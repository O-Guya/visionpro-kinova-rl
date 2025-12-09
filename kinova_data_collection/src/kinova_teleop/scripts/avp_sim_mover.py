#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import copy
import time
import threading
import math
import os
import signal
import numpy as np
import moveit_commander
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation as R
from avp_stream import VisionProStreamer

try:
    from trac_ik_python.trac_ik import IK
except ImportError:
    print("请先安装 trac_ik: sudo apt-get install ros-noetic-trac-ik-python")
    sys.exit(1)


# ==========================================
# 1. AVP Bridge (保持不变)
# ==========================================
class AVPBridge:
    def __init__(self, target_ip=None):
        if target_ip:
            self.avp_ip = target_ip
        else:
            self.avp_ip = rospy.get_param("~avp_ip", "192.168.1.50")

        self.pose_pub = rospy.Publisher(
            "/vision_pro/right_hand_pose", PoseStamped, queue_size=1
        )
        self.raw_pose_pub = rospy.Publisher(
            "/vision_pro/right_hand_pose_raw", PoseStamped, queue_size=1
        )
        self.pinch_pub = rospy.Publisher(
            "/vision_pro/left_pinch", Float32, queue_size=1
        )

        self.transform_matrix = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        self.ALPHA = 0.3  # 滤波系数 (0.1 更平滑, 0.5 更跟手)
        self.filtered_position = None
        self.filtered_orientation = None

        self.keep_running = False
        self.bridge_thread = None
        self.streamer = None
        self.data_lock = threading.Lock()

    def start(self):
        if self.keep_running:
            return
        self.keep_running = True
        self.bridge_thread = threading.Thread(target=self._bridge_worker, daemon=True)
        self.bridge_thread.start()
        rospy.loginfo("[AVPBridge] 后台线程已启动...")

    def stop(self):
        self.keep_running = False

    def _bridge_worker(self):
        while self.keep_running and not rospy.is_shutdown():
            try:
                if self.streamer is None:
                    try:
                        new_streamer = VisionProStreamer(ip=self.avp_ip, record=False)
                        with self.data_lock:
                            self.streamer = new_streamer
                        rospy.loginfo("[AVPBridge] 连接成功！")
                    except Exception:
                        time.sleep(2.0)
                        continue

                if self.streamer:
                    data = None
                    try:
                        data = self.streamer.latest
                    except Exception:
                        pass

                    if data:
                        self._process_data(data)
                    time.sleep(0.01)

            except Exception:
                with self.data_lock:
                    self.streamer = None
                time.sleep(1.0)

    def _process_data(self, data):
        try:
            if "right_wrist" in data:
                matrix_avp = np.array(data["right_wrist"])

                if matrix_avp.ndim == 3:
                    matrix_avp = matrix_avp[0]
                if matrix_avp.shape == (4, 4):
                    # 坐标变换
                    matrix_kinova = np.dot(self.transform_matrix, matrix_avp)

                    # 提取原始 P/Q
                    raw_pos = matrix_kinova[:3, 3]
                    raw_quat = R.from_matrix(matrix_kinova[:3, :3]).as_quat()

                    # 发布 raw 数据
                    raw_msg = PoseStamped()
                    raw_msg.header.stamp = rospy.Time.now()
                    raw_msg.header.frame_id = "world"
                    raw_msg.pose.position.x = raw_pos[0]
                    raw_msg.pose.position.y = raw_pos[1]
                    raw_msg.pose.position.z = raw_pos[2]
                    raw_msg.pose.orientation.x = raw_quat[0]
                    raw_msg.pose.orientation.y = raw_quat[1]
                    raw_msg.pose.orientation.z = raw_quat[2]
                    raw_msg.pose.orientation.w = raw_quat[3]
                    # 发布原始数据
                    self.raw_pose_pub.publish(raw_msg)

                    # 4. EMA 滤波
                    if self.filtered_position is None:
                        self.filtered_position = raw_pos
                        self.filtered_orientation = raw_quat
                    else:
                        alpha = self.ALPHA
                        self.filtered_position = (
                            alpha * raw_pos + (1 - alpha) * self.filtered_position
                        )
                        q_new = (
                            alpha * raw_quat + (1 - alpha) * self.filtered_orientation
                        )
                        q_new /= np.linalg.norm(q_new)
                        self.filtered_orientation = q_new

                    # 5. 发布滤波后数据
                    msg = PoseStamped()
                    msg.header.stamp = rospy.Time.now()
                    msg.header.frame_id = "world"
                    msg.pose.position.x = self.filtered_position[0]
                    msg.pose.position.y = self.filtered_position[1]
                    msg.pose.position.z = self.filtered_position[2]
                    msg.pose.orientation.x = self.filtered_orientation[0]
                    msg.pose.orientation.y = self.filtered_orientation[1]
                    msg.pose.orientation.z = self.filtered_orientation[2]
                    msg.pose.orientation.w = self.filtered_orientation[3]
                    self.pose_pub.publish(msg)

            if "left_pinch_distance" in data:
                self.pinch_pub.publish(float(data["left_pinch_distance"]))
        except Exception:
            pass


# ==========================================
# 2. AVPMover (Trac-IK 极速版)
# ==========================================
class AVPMover:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.on_shutdown(self.shutdown)

        self.bridge = AVPBridge()
        self.bridge.start()

        self.group = moveit_commander.MoveGroupCommander("arm")
        self.fake_joint_pub = rospy.Publisher(
            "/move_group/fake_controller_joint_states", JointState, queue_size=1
        )

        # === 1. 初始化 Trac-IK ===
        # 它会自动从 ROS 参数服务器读取 /robot_description，不需要手动指定路径
        rospy.loginfo("Initializing Trac-IK solver...")
        try:
            # base_link 和 tip_link 必须和你 URDF 里的一致
            # 对于 j2s7s300，通常是 'j2s7s300_link_base' 和 'j2s7s300_end_effector'
            self.ik_solver = IK(
                "j2s7s300_link_base", "j2s7s300_end_effector", timeout=0.005
            )
            rospy.loginfo(f"Trac-IK Ready! Joint names: {self.ik_solver.joint_names}")
        except Exception as e:
            rospy.logerr(f"Trac-IK Init Failed: {e}")
            sys.exit(1)

        # 变量初始化
        self.lock = threading.Lock()

        # 确保关节名字顺序和 IK 解算器一致
        self.arm_joint_names = list(self.ik_solver.joint_names)
        self.finger_joint_names = [
            "j2s7s300_joint_finger_1",
            "j2s7s300_joint_finger_2",
            "j2s7s300_joint_finger_3",
        ]

        self.current_joint_values = [0.0] * 7
        self.current_finger_values = [0.0] * 3

        # 目标状态
        self.target_joint_values = None
        self.target_finger_values = None

        self.robot_start_pose = None
        self.hand_start_pose = None
        self.latest_hand_pose = None
        self.scale_factor = 1.0
        self.pinch_val = 1.0

        # === 启动显示线程 (60Hz 平滑刷新) ===
        self.keep_running = True
        self.pub_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.pub_thread.start()

        # 复位
        self.go_to_ready_pose()

        print("\n" + "=" * 40)
        print("Ready Pose Reached.")
        print("Press ENTER to UNLOCK simulation...")
        print("=" * 40 + "\n")
        input()

        # 解锁
        self.robot_start_pose = self.group.get_current_pose().pose
        self.hand_start_pose = None
        self.control_enabled = True

        rospy.loginfo("Control UNLOCKED! High Performance Mode.")

        self.pinch_sub = rospy.Subscriber(
            "/vision_pro/left_pinch", Float32, self.pinch_callback
        )
        self.sub = rospy.Subscriber(
            "/vision_pro/right_hand_pose",
            PoseStamped,
            self.callback,
            queue_size=1,
            buff_size=2**24,
        )

        # === 启动计算线程 ===
        self.ik_thread = threading.Thread(target=self.ik_loop, daemon=True)
        self.ik_thread.start()

    def shutdown(self):
        rospy.logwarn("Shutting down...")
        self.keep_running = False
        self.bridge.stop()
        self.group.stop()
        os.kill(os.getpid(), signal.SIGKILL)

    # === 显示线程：只负责平滑插值和发布 ===
    def display_loop(self):
        rate = rospy.Rate(60)
        SMOOTH_FACTOR = 0.3  # 越大越跟手，越小越平滑

        while self.keep_running and not rospy.is_shutdown():
            try:
                # 1. 读目标
                target_j = None
                target_f = None
                with self.lock:
                    if self.target_joint_values:
                        target_j = list(self.target_joint_values)
                        target_f = list(self.target_finger_values)

                # 2. 插值
                if target_j:
                    new_joints = []
                    for curr, tgt in zip(self.current_joint_values, target_j):
                        new_joints.append(curr + (tgt - curr) * SMOOTH_FACTOR)
                    self.current_joint_values = new_joints

                    if target_f:
                        new_fingers = []
                        for curr, tgt in zip(self.current_finger_values, target_f):
                            new_fingers.append(curr + (tgt - curr) * SMOOTH_FACTOR)
                        self.current_finger_values = new_fingers

                # 3. 发布
                js_msg = JointState()
                js_msg.header.stamp = rospy.Time.now()
                js_msg.name = self.arm_joint_names + self.finger_joint_names
                js_msg.position = list(self.current_joint_values) + list(
                    self.current_finger_values
                )
                self.fake_joint_pub.publish(js_msg)
            except Exception:
                pass
            rate.sleep()

    # === 计算线程：IK 计算 (极速) ===
    def ik_loop(self):
        rate = rospy.Rate(100)
        while self.keep_running and not rospy.is_shutdown():
            try:
                current_hand = None
                with self.lock:
                    if self.latest_hand_pose:
                        current_hand = copy.deepcopy(self.latest_hand_pose)

                if current_hand is None or self.hand_start_pose is None:
                    time.sleep(0.01)
                    continue

                # Pose
                dx = current_hand.position.x - self.hand_start_pose.position.x
                dy = current_hand.position.y - self.hand_start_pose.position.y
                dz = current_hand.position.z - self.hand_start_pose.position.z

                target_pose = copy.deepcopy(self.robot_start_pose)
                target_pose.position.x += dx * self.scale_factor
                target_pose.position.y += dy * self.scale_factor
                target_pose.position.z += dz * self.scale_factor

                # Rot
                try:
                    q_hand_curr = [
                        current_hand.orientation.x,
                        current_hand.orientation.y,
                        current_hand.orientation.z,
                        current_hand.orientation.w,
                    ]
                    q_hand_start = [
                        self.hand_start_pose.orientation.x,
                        self.hand_start_pose.orientation.y,
                        self.hand_start_pose.orientation.z,
                        self.hand_start_pose.orientation.w,
                    ]
                    q_robot_start = [
                        self.robot_start_pose.orientation.x,
                        self.robot_start_pose.orientation.y,
                        self.robot_start_pose.orientation.z,
                        self.robot_start_pose.orientation.w,
                    ]
                    r_delta = R.from_quat(q_hand_curr) * R.from_quat(q_hand_start).inv()
                    r_target = r_delta * R.from_quat(q_robot_start)
                    q_final = r_target.as_quat()
                    target_pose.orientation.x = q_final[0]
                    target_pose.orientation.y = q_final[1]
                    target_pose.orientation.z = q_final[2]
                    target_pose.orientation.w = q_final[3]
                except:
                    pass

                # Limit
                if target_pose.position.z < 0.05:
                    target_pose.position.z = 0.05
                if target_pose.position.x > 0.8:
                    target_pose.position.x = 0.8

                # === Trac-IK 解算 ===
                # 必须提供 seed (当前关节角)
                # 使用本地缓存的关节角作为 seed
                with self.lock:
                    seed_state = list(self.current_joint_values)

                # trac_ik 需要 x, y, z, qx, qy, qz, qw
                sol = self.ik_solver.get_ik(
                    seed_state,
                    target_pose.position.x,
                    target_pose.position.y,
                    target_pose.position.z,
                    target_pose.orientation.x,
                    target_pose.orientation.y,
                    target_pose.orientation.z,
                    target_pose.orientation.w,
                )

                if sol:
                    gripper = self.get_gripper_angle()
                    with self.lock:
                        self.target_joint_values = sol
                        self.target_finger_values = [gripper] * 3

                rate.sleep()
            except Exception:
                pass

    def go_to_ready_pose(self):
        rospy.loginfo("Moving to Ready Pose...")
        target_joints = [4.71, 3.14, 6.28, 0.79, 0.0, 3.93, 4.71]
        # 插值复位
        start_joints = list(self.current_joint_values)
        steps = 60
        for i in range(steps):
            alpha = (i + 1) / steps
            interp = []
            for s, e in zip(start_joints, target_joints):
                interp.append(s + (e - s) * alpha)
            with self.lock:
                self.current_joint_values = interp
            time.sleep(0.02)

        with self.lock:
            self.current_joint_values = target_joints

        try:
            self.group.remember_joint_values("ready", target_joints)
            self.robot_start_pose = self.group.get_current_pose().pose
            rospy.loginfo("Ready Pose Done.")
        except:
            pass

    def pinch_callback(self, msg):
        self.pinch_val = msg.data

    def get_gripper_angle(self):
        dist = max(0.0, min(self.pinch_val, 0.08))
        ratio = dist / 0.08
        return (1.0 - ratio) * 1.3

    def callback(self, msg):
        with self.lock:
            if self.hand_start_pose is None:
                self.hand_start_pose = msg.pose
                rospy.loginfo(">>> Zero Point Locked!")
            self.latest_hand_pose = msg.pose


# ==========================================
# 3. AVP Tuner 滤波器调参及可视化类
# ==========================================
class AVPTuner:
    """负责将 Pose 拆解为独立的坐标分量，方便 rqt_plot 对比波形"""

    def __init__(self):
        # 1. 启动 Bridge
        self.bridge = AVPBridge()
        self.bridge.start()
        rospy.on_shutdown(self.shutdown)

        # 2. 【修改】创建 X/Y/Z 的对比话题
        # 这里的命名方式是为了让 rqt_plot 的图例更清晰
        self.pub_raw_x = rospy.Publisher("/tuner/x/raw", Float32, queue_size=1)
        self.pub_filt_x = rospy.Publisher("/tuner/x/filtered", Float32, queue_size=1)

        self.pub_raw_y = rospy.Publisher("/tuner/y/raw", Float32, queue_size=1)
        self.pub_filt_y = rospy.Publisher("/tuner/y/filtered", Float32, queue_size=1)

        self.pub_raw_z = rospy.Publisher("/tuner/z/raw", Float32, queue_size=1)
        self.pub_filt_z = rospy.Publisher("/tuner/z/filtered", Float32, queue_size=1)

        # 3. 订阅 Bridge 数据
        rospy.Subscriber("/vision_pro/right_hand_pose_raw", PoseStamped, self._raw_cb)
        rospy.Subscriber("/vision_pro/right_hand_pose", PoseStamped, self._filtered_cb)

        # 数据缓存
        self.raw_pose = None
        self.filtered_pose = None

        rospy.loginfo("Filter Tuner Mode ACTIVE.")
        rospy.loginfo("Run 'rqt_plot' command provided in instructions to visualize.")

    def shutdown(self):
        self.bridge.stop()

    def _raw_cb(self, msg):
        self.raw_pose = msg
        self._publish_metrics()

    def _filtered_cb(self, msg):
        self.filtered_pose = msg
        # 等待 raw 更新后再统一发布，保持时间轴对齐

    def _publish_metrics(self):
        """拆解并发布分量"""
        if self.raw_pose is None or self.filtered_pose is None:
            return

        # 发布 X 轴对比
        self.pub_raw_x.publish(self.raw_pose.pose.position.x)
        self.pub_filt_x.publish(self.filtered_pose.pose.position.x)

        # 发布 Y 轴对比
        self.pub_raw_y.publish(self.raw_pose.pose.position.y)
        self.pub_filt_y.publish(self.filtered_pose.pose.position.y)

        # 发布 Z 轴对比
        self.pub_raw_z.publish(self.raw_pose.pose.position.z)
        self.pub_filt_z.publish(self.filtered_pose.pose.position.z)

        # 可选：打印简单的状态保活
        # sys.stdout.write("\r[Tuner] Publishing split streams for RQT...")
        # sys.stdout.flush()

    def run(self):
        rospy.spin()


def main_executor():
    # 检查是否处于调参模式 (通过命令行参数 'tuner' 判断)
    is_tuning_mode = "tuner" in sys.argv

    # 必须先初始化 ROS Master
    rospy.init_node("avp_main_executor", anonymous=True)

    if is_tuning_mode:
        # 模式 A: 调参模式
        tuner = AVPTuner()
        tuner.run()
    else:
        # 模式 B: 控制模式
        moveit_commander.roscpp_initialize(sys.argv)
        mover = AVPMover()
        rospy.spin()


if __name__ == "__main__":
    try:
        main_executor()
    except rospy.ROSInterruptException:
        pass
