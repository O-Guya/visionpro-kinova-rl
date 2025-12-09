#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys
import copy
import threading
import time
import numpy as np
import math
import moveit_commander
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float32MultiArray
from kinova_msgs.msg import JointVelocityWithFingerVelocity
from kinova_msgs.srv import Start, Stop

from avp_sim_mover import AVPBridge


# === 辅助类：OneEuroFilter ===
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class KinovaBaseMover:
    """
    【基础类】KinovaBaseMover
    功能：
    1. 关节伺服控制 (PID)
    2. 末端笛卡尔控制 (集成 IK 解算)
    """

    def __init__(self):
        try:
            rospy.get_rostime()
        except rospy.ROSInitException:
            moveit_commander.roscpp_initialize(sys.argv)
            rospy.init_node("kinova_driver_node", anonymous=True)

        rospy.on_shutdown(self.shutdown)

        self.group = moveit_commander.MoveGroupCommander("arm")
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper")

        # === 1. 控制参数 ===
        self.KP_ARM = 80.0
        self.KI_ARM = 0.0
        self.I_LIMIT = 10.0
        self.JOINT_MAX_VEL = 35.0
        self.FINGER_VEL_CLOSE = 6000.0
        self.FINGER_VEL_OPEN = -6000.0

        # 复位姿态
        self.rest_joint_values = np.array([4.71, 3.14, 6.28, 0.79, 0.0, 3.93, 4.71])
        # 权重用于 IK 偏置 (倾向于保持在此姿态附近)
        self.rest_joint_weights = np.array([0.02, 0.02, 0.30, 0.02, 0.02, 0.02, 0.02])

        # === 2. IK 初始化  ===
        try:
            from trac_ik_python.trac_ik import IK

            self.ik_solver = IK(
                "j2s7s300_link_base", "j2s7s300_end_effector", timeout=0.03
            )
            self.ik_available = True
        except ImportError:
            rospy.logwarn("[BaseMover] TracIK not found! Cartesian control will fail.")
            self.ik_available = False

        self.JUMP_THRESHOLDS = np.array([1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0])

        # 关节平滑滤波器
        self.filters = []
        for _ in range(7):
            self.filters.append(
                OneEuroFilter(time.time(), 0.0, min_cutoff=0.5, beta=0.1)
            )

        # === 3. 状态变量 ===
        self.lock = threading.Lock()
        self.current_joint_values = [0.0] * 7
        self.current_finger_values = [0.0] * 3
        self.current_joint_velocities = [0.0] * 7
        self.current_joint_efforts = [0.0] * 7

        # 目标变量

        self.target_joints = None  # [J1...J7]
        self.target_finger = 0.0
        self.control_active = False
        self.integral_error = [0.0] * 7

        # 笛卡尔缓存
        self.tool_pose_lock = threading.Lock()
        self.current_tool_pose = None
        self.target_cartesian_pose = None

        # === 4. 通信 ===
        self.velocity_pub = rospy.Publisher(
            "/j2s7s300_driver/in/joint_velocity_with_finger_velocity",
            JointVelocityWithFingerVelocity,
            queue_size=1,
        )
        self.joint_sub = rospy.Subscriber(
            "/j2s7s300_driver/out/joint_state",
            JointState,
            self.joint_cb,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.tool_pose_sub = rospy.Subscriber(
            "/j2s7s300_driver/out/tool_pose",
            PoseStamped,
            self.tool_pose_cb,
            queue_size=1,
        )
        self.debug_target_pub = rospy.Publisher(
            "/debug/joint/target", Float32MultiArray, queue_size=1
        )
        self.debug_current_pub = rospy.Publisher(
            "/debug/joint/current", Float32MultiArray, queue_size=1
        )

        # 启动 PID 线程
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()

        self.go_to_ready_pose()

    def shutdown(self):
        self.control_active = False
        try:
            self.velocity_pub.publish(JointVelocityWithFingerVelocity())
        except:
            pass

    # --- Callbacks ---
    def joint_cb(self, msg):
        if len(msg.position) >= 7:
            with self.lock:
                self.current_joint_values = list(msg.position[:7])
                if len(msg.velocity) >= 7:
                    self.current_joint_velocities = list(msg.velocity[:7])
                if len(msg.effort) >= 7:
                    self.current_joint_efforts = list(msg.effort[:7])
                if len(msg.position) >= 10:
                    self.current_finger_values = list(msg.position[7:10])

    def tool_pose_cb(self, msg):
        with self.tool_pose_lock:
            self.current_tool_pose = msg

    # --- Control Interface ---
    def enable_control(self):
        with self.lock:
            self.target_joints = list(self.current_joint_values)  # Lock current
            self.integral_error = [0.0] * 7
            self.control_active = True
        rospy.loginfo(">>> BaseMover: Control ENABLED")
        return True

    def disable_control(self):
        self.control_active = False
        rospy.loginfo(">>> BaseMover: Control DISABLED")

    def set_target_joints(self, joints, finger_state):
        """直接设定关节角度"""
        with self.lock:
            self.target_joints = list(joints)
            self.target_finger = float(finger_state)

    def set_target_cartesian(self, position, orientation, finger_state):
        """
        设定末端位姿 (集成 IK)
        position: [x, y, z]
        orientation: [qx, qy, qz, qw]
        """
        if not self.ik_available:
            return False

        # [修复] 在这里保存目标位姿，以便 Getter 读取
        # 构造一个简单的 Pose 对象存储，以便 get_target_cartesian 使用
        p = Pose()
        p.position.x = position[0]
        p.position.y = position[1]
        p.position.z = position[2]
        p.orientation.x = orientation[0]
        p.orientation.y = orientation[1]
        p.orientation.z = orientation[2]
        p.orientation.w = orientation[3]

        with self.lock:
            self.target_cartesian_pose = p

        # 1. 准备 IK 种子 (Bias towards rest pose)
        def unwrap_angle(source, target):
            diff = target - source
            turns = np.round(diff / (2 * np.pi))
            return target - turns * 2 * np.pi

        with self.lock:
            curr_array = np.array(self.current_joint_values)

        aligned_rest = np.zeros(7)
        for i in range(7):
            aligned_rest[i] = unwrap_angle(curr_array[i], self.rest_joint_values[i])

        biased_seed = (
            curr_array * (1.0 - self.rest_joint_weights)
            + aligned_rest * self.rest_joint_weights
        )
        seed_state = list(biased_seed)

        # 2. 解算 IK
        sol = self.ik_solver.get_ik(
            seed_state,
            position[0],
            position[1],
            position[2],
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
            0.01,
            0.01,
            0.01,
            0.1,
            0.1,
            0.1,  # Tolerances
        )

        if sol:
            raw_sol = np.array(sol)
            # 3. 安全检查 (Jump Rejection)
            if np.any(np.abs(raw_sol - curr_array) > self.JUMP_THRESHOLDS):
                return False  # 跳变过大，忽略

            # 4. 滤波
            now = time.time()
            filtered_sol = []
            for i in range(7):
                filtered_sol.append(self.filters[i](now, raw_sol[i]))

            # 5. 设置目标
            self.set_target_joints(filtered_sol, finger_state)
            return True
        else:
            return False

    def _get_gripper_mean(self):
        """内部辅助：获取夹爪平均值"""
        fingers = list(self.current_finger_values)
        return np.mean(fingers) if fingers else 0.0

    def get_current_joints(self):
        """
        获取当前关节状态 (包含夹爪)
        Returns:
            list[float]: [j1, j2, j3, j4, j5, j6, j7, gripper_val] (8维)
        """
        with self.lock:
            joints = list(self.current_joint_values)
            g_val = self._get_gripper_mean()
        return joints + [g_val]

    def get_current_cartesian(self):
        """
        获取当前笛卡尔状态 (包含夹爪)
        Returns:
            list[float]: [x, y, z, rx, ry, rz, gripper_val] (7维)
        """
        pose_msg = None
        with self.tool_pose_lock:
            if self.current_tool_pose:
                pose_msg = copy.deepcopy(self.current_tool_pose.pose)

        with self.lock:
            g_val = self._get_gripper_mean()

        if pose_msg is None:
            return [0.0] * 6 + [g_val]

        pos = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
        try:
            r = R.from_quat(
                [
                    pose_msg.orientation.x,
                    pose_msg.orientation.y,
                    pose_msg.orientation.z,
                    pose_msg.orientation.w,
                ]
            )
            eulers = r.as_euler("xyz", degrees=False)
            return list(pos) + list(eulers) + [g_val]
        except Exception:
            return [0.0] * 6 + [g_val]

    def get_target_joints(self):
        """
        获取当前的控制目标 (包含夹爪)
        Returns:
            list[float]: [t_j1...t_j7, t_gripper]
        """
        with self.lock:
            if self.target_joints is None:
                # 如果还没设置过目标，返回当前状态作为默认目标
                return self.get_current_joints()

            t_joints = list(self.target_joints)
            t_finger = self.target_finger
        return t_joints + [t_finger]

    def get_target_cartesian(self):
        """
        获取当前的笛卡尔目标 (包含夹爪)
        注意：如果没有显式调用 set_target_cartesian，这里返回的可能是 None 或上次计算的值
        Returns:
            list[float]: [t_x, t_y, t_z, t_rx, t_ry, t_rz, t_gripper]
        """
        # 如果是 set_target_cartesian 设置的，我们可能有缓存
        # 如果没有缓存（比如是纯关节控制模式），这里只能返回当前笛卡尔位置
        if self.target_cartesian_pose:
            p = self.target_cartesian_pose
            pos = [p.position.x, p.position.y, p.position.z]
            r = R.from_quat(
                [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
            )

            with self.lock:
                t_finger = self.target_finger

            return list(pos) + list(r.as_euler("xyz", degrees=False)) + [t_finger]

        # Fallback
        return self.get_current_cartesian()

    def go_to_ready_pose(self):
        self.control_active = False
        try:
            self.group.set_max_velocity_scaling_factor(1.0)
            self.group.go(list(self.rest_joint_values), wait=True)
            self.group.stop()
            try:
                start_srv = rospy.ServiceProxy("/j2s7s300_driver/in/start", Start)
                start_srv()
            except:
                pass
            self.integral_error = [0.0] * 7
            rospy.loginfo(">>> Robot Ready.")
        except Exception as e:
            print(e)

    def control_loop(self):
        """PID 循环，只负责追随 self.target_joints"""
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self.control_active:
                try:
                    target_j, curr_j = None, None
                    t_finger = 0.0
                    with self.lock:
                        if self.target_joints:
                            target_j = list(self.target_joints)
                        curr_j = list(self.current_joint_values)
                        t_finger = self.target_finger

                    if target_j and curr_j:
                        vel_msg = JointVelocityWithFingerVelocity()
                        cmds = []
                        for i in range(7):
                            err = target_j[i] - curr_j[i]
                            self.integral_error[i] += err
                            self.integral_error[i] = max(
                                min(self.integral_error[i], self.I_LIMIT), -self.I_LIMIT
                            )
                            raw_v = (
                                err * self.KP_ARM + self.integral_error[i] * self.KI_ARM
                            )
                            cmds.append(
                                max(min(raw_v, self.JOINT_MAX_VEL), -self.JOINT_MAX_VEL)
                            )

                        f_cmd = (
                            self.FINGER_VEL_CLOSE
                            if t_finger > 0.5
                            else self.FINGER_VEL_OPEN
                        )

                        (
                            vel_msg.joint1,
                            vel_msg.joint2,
                            vel_msg.joint3,
                            vel_msg.joint4,
                            vel_msg.joint5,
                            vel_msg.joint6,
                            vel_msg.joint7,
                        ) = cmds
                        vel_msg.finger1 = vel_msg.finger2 = vel_msg.finger3 = f_cmd
                        self.velocity_pub.publish(vel_msg)

                        # Debug
                        self.debug_target_pub.publish(Float32MultiArray(data=target_j))
                        self.debug_current_pub.publish(Float32MultiArray(data=curr_j))
                except:
                    pass
            rate.sleep()


class AVPTeleopMover(KinovaBaseMover):
    """
    【遥操类】AVPTeleopMover
    功能：
    1. 接收 Vision Pro 数据
    2. 计算相对位移 (Delta Pose)
    3. 调用基类 set_target_cartesian() 进行解算和控制
    """

    def __init__(self, standalone=True):
        # 1. 初始化基类 (包含 IK)
        super().__init__()

        if AVPBridge is None:
            rospy.logerr("AVPBridge not found!")
            sys.exit(1)

        self.bridge = AVPBridge()
        self.bridge.start()

        # 参数
        self.ENABLE_ORIENTATION = True
        self.limit_x = (-0.7, 0.7)
        self.limit_y = (-0.7, 0.7)
        self.limit_z = (-0.05, 1.0)
        self.scale_factor = 1.0

        # 状态
        self.pinch_val = 1.0
        self.robot_start_pose = None
        self.hand_start_pose = None
        self.latest_hand_pose = None

        self.pinch_sub = rospy.Subscriber(
            "/vision_pro/left_pinch", Float32, self.pinch_cb
        )
        self.sub = rospy.Subscriber(
            "/vision_pro/right_hand_pose", PoseStamped, self.hand_cb, queue_size=1
        )

        # 启动计算线程 (不再是 IK loop，而是 Pose loop)
        self.pose_loop_thread = threading.Thread(target=self.pose_calculation_loop)
        self.pose_loop_thread.start()

        if standalone:
            self.wait_for_pinch_unlock()
        else:
            rospy.loginfo("[AVPTeleop] Passive Mode Ready.")

    def shutdown(self):
        super().shutdown()
        try:
            self.bridge.stop()
        except:
            pass

    def pinch_cb(self, msg):
        self.pinch_val = msg.data

    def hand_cb(self, msg):
        with self.lock:
            self.latest_hand_pose = msg.pose

    def wait_for_pinch_unlock(self):
        # 1. 第一阶段：静默等待连接和数据
        rospy.loginfo("Waiting for Vision Pro connection & data stream...")

        # 循环等待直到收到第一帧 Pose 数据
        while self.latest_hand_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # 收到数据后，稍微等一下让数据流稳定，防止瞬时噪声
        rospy.sleep(0.5)

        # 2. 第二阶段：打印 UI
        print("=" * 60)
        print("✅  VISION PRO CONNECTED  ✅")
        print("=" * 60)
        print("⚠️   STANDALONE MODE   ⚠️")
        print("【控制方式】：左手捏合，右手移动")
        print("【解锁方式】：捏合持续 3 秒...")
        print("=" * 60 + "\n")

        pinch_start_time = None

        # 定义清行函数，用于刷新进度条
        def clear_line():
            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()

        # 3. 第三阶段：解锁循环
        while not rospy.is_shutdown():
            # 捏合判定
            if self.pinch_val < 0.03:
                if pinch_start_time is None:
                    pinch_start_time = rospy.Time.now()

                duration = (rospy.Time.now() - pinch_start_time).to_sec()
                display_duration = min(duration, 3.0)

                # 进度条动画
                bar_len = 20
                filled_len = int(display_duration / 3.0 * bar_len)
                bar = "█" * filled_len + "░" * (bar_len - filled_len)

                sys.stdout.write(
                    f"\r🔓 Unlocking: [{bar}] {display_duration:.1f}s / 3.0s"
                )
                sys.stdout.flush()

                if duration >= 3.0:
                    clear_line()
                    print(f"\r>>> ✅ UNLOCKED! Control Enabled.")
                    self.enable_control()
                    break
            else:
                # 松手重置
                if pinch_start_time is not None:
                    clear_line()
                    sys.stdout.write(f"\r>>> ⏳ Waiting for pinch...")
                    sys.stdout.flush()
                pinch_start_time = None

            rospy.sleep(0.05)

    def enable_control(self):
        if self.latest_hand_pose is None:
            return False
        with self.lock:
            self.robot_start_pose = self.group.get_current_pose().pose
            self.hand_start_pose = copy.deepcopy(self.latest_hand_pose)
        return super().enable_control()

    def pose_calculation_loop(self):
        """计算目标 Pose 并发送给基类"""

        def clamp(val, limits):
            return max(limits[0], min(val, limits[1]))

        while not rospy.is_shutdown():
            if not self.control_active:
                time.sleep(0.01)
                continue

            try:
                # 1. 计算目标位姿
                with self.lock:
                    curr_hand = copy.deepcopy(self.latest_hand_pose)

                if curr_hand is None or self.hand_start_pose is None:
                    continue

                # Delta Position
                dx = curr_hand.position.x - self.hand_start_pose.position.x
                dy = curr_hand.position.y - self.hand_start_pose.position.y
                dz = curr_hand.position.z - self.hand_start_pose.position.z

                target_pos = [
                    clamp(
                        self.robot_start_pose.position.x + dx * self.scale_factor,
                        self.limit_x,
                    ),
                    clamp(
                        self.robot_start_pose.position.y + dy * self.scale_factor,
                        self.limit_y,
                    ),
                    clamp(
                        self.robot_start_pose.position.z + dz * self.scale_factor,
                        self.limit_z,
                    ),
                ]

                # Delta Orientation
                target_quat = [
                    self.robot_start_pose.orientation.x,
                    self.robot_start_pose.orientation.y,
                    self.robot_start_pose.orientation.z,
                    self.robot_start_pose.orientation.w,
                ]

                if self.ENABLE_ORIENTATION:
                    try:
                        q_h_c = [
                            curr_hand.orientation.x,
                            curr_hand.orientation.y,
                            curr_hand.orientation.z,
                            curr_hand.orientation.w,
                        ]
                        q_h_s = [
                            self.hand_start_pose.orientation.x,
                            self.hand_start_pose.orientation.y,
                            self.hand_start_pose.orientation.z,
                            self.hand_start_pose.orientation.w,
                        ]

                        r_delta = R.from_quat(q_h_c) * R.from_quat(q_h_s).inv()
                        r_target = r_delta * R.from_quat(target_quat)
                        target_quat = r_target.as_quat()
                    except:
                        pass

                # 2. 夹爪逻辑
                f_state = 1.0 if self.pinch_val < 0.03 else 0.0

                # 3. 核心修改：调用基类方法进行 IK 和控制
                self.set_target_cartesian(target_pos, target_quat, f_state)

                time.sleep(0.002)
            except Exception as e:
                pass


if __name__ == "__main__":
    AVPTeleopMover(standalone=True)
