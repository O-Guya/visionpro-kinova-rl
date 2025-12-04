import numpy as np
import gymnasium as gym
import time
import copy
import threading
from collections import OrderedDict
from scipy.spatial.transform import Rotation

# === ROS 2 依赖 ===
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from control_msgs.action import GripperCommand
from tf2_ros import Buffer, TransformListener

# === 引入相机模块 ===
from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
# 假设你已经按上一条回复建好了这个文件
from franka_env.camera.webcam_capture import WebcamCapture 
from franka_env.envs.franka_env import DefaultEnvConfig # 复用配置类

# =============================================================================
# 1. ROS 桥接节点 (负责所有脏活累活)
# =============================================================================
class KinovaBridge(Node):
    def __init__(self):
        super().__init__('kinova_rl_bridge')
        
        # --- A. 机械臂控制 (发布给笛卡尔阻抗控制器) ---
        self.target_pub = self.create_publisher(
            PoseStamped, 
            '/my_compliance_controller/target_frame', # 对应 yaml 里的配置
            10
        )
        
        # --- B. 夹爪控制 (Action Client) ---
        self._gripper_client = ActionClient(
            self, 
            GripperCommand, 
            '/robotiq_2f_85_gripper_controller/gripper_cmd' # 需用 ros2 action list 确认
        )

        # --- C. 状态监听 (TF + Joints) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.latest_joint_state = None
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_cb, 10
        )
        
        # --- D. 启动独立线程处理 ROS 回调 ---
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        self.get_logger().info("Kinova Bridge Started!")

    def _spin(self):
        rclpy.spin(self)

    def joint_cb(self, msg):
        self.latest_joint_state = msg

    def get_current_pose(self):
        """从 TF 获取当前末端位姿 [x,y,z, qx,qy,qz,qw]"""
        try:
            # 查 TF: base_link -> tool_frame (名字要和 URDF 一致)
            t = self.tf_buffer.lookup_transform(
                'base_link', 'gen3_end_effector_link', rclpy.time.Time()
            )
            pos = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            rot = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            return np.concatenate([pos, rot])
        except Exception as e:
            # 刚启动时可能查不到 TF，返回 None 或 0
            return np.zeros(7)

    def publish_target(self, pos):
        """pos: [x,y,z, qx,qy,qz,qw]"""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(pos[3])
        msg.pose.orientation.y = float(pos[4])
        msg.pose.orientation.z = float(pos[5])
        msg.pose.orientation.w = float(pos[6])
        self.target_pub.publish(msg)

    def move_gripper(self, width):
        """阻塞式控制夹爪"""
        goal = GripperCommand.Goal()
        goal.command.position = width # 0.0 (开) - 0.8 (闭，需测试)
        goal.command.max_effort = 100.0
        
        future = self._gripper_client.send_goal_async(goal)
        # 这里为了简单，我们不完全阻塞等待结果，只发送指令
        # 如果需要严格同步，参考之前的 Action 代码
        # 但 RL 训练中通常希望 step 尽快返回，所以这里 fire-and-forget 也是一种选择
        # 为了严谨，建议加个微小的 sleep 在 env 里

# =============================================================================
# 2. Kinova 环境类 (继承 Gym)
# =============================================================================
class KinovaEnv(gym.Env):
    def __init__(self, config: DefaultEnvConfig, **kwargs):
        # 初始化 ROS
        if not rclpy.ok():
            rclpy.init()
        self.bridge = KinovaBridge()
        
        # 复制配置
        self.config = config
        self.action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        # ... (其他基础变量复制 FrankaEnv 的代码) ...
        
        # ⚠️ 关键修改：相机初始化 (使用混合配置)
        self.cap = None
        self.init_cameras(config.CAMERAS_CONFIG) # 记得改 config 里的结构
        
        # 定义空间 (同 Franka)
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)), # 暂时留空
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)), # 暂时留空
            }),
            # ... 图片空间 ...
        })
        self.action_space = gym.spaces.Box(-1, 1, shape=(7,))
        
        # 初始位置读取
        time.sleep(1) # 等待 TF 建立
        self._update_currpos()
        print("Kinova Env Initialized!")

    def init_cameras(self, camera_config):
        # ... (把上一条回答里的 init_cameras 代码粘贴到这里) ...
        pass

    def step(self, action):
        # 1. 计算目标位姿 (和 FrankaEnv 逻辑一样，计算 delta)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]
        
        self.nextpos = self.currpos.copy()
        self.nextpos[:3] += xyz_delta * self.action_scale[0]
        
        # 计算旋转 (四元数运算)
        # 注意：scipy 的 Rotation 顺序可能有坑，保持和 FrankaEnv 一致即可
        rot_action = Rotation.from_rotvec(action[3:6] * self.action_scale[1])
        curr_rot = Rotation.from_quat(self.currpos[3:])
        self.nextpos[3:] = (rot_action * curr_rot).as_quat()

        # 2. 安全裁切 (Safety Box) - 必须重写 clip_safety_box 里的尺寸
        self.nextpos = self.clip_safety_box(self.nextpos)

        # 3. 发送指令 (ROS)
        # A. 夹爪
        gripper_action = action[6]
        self._send_gripper_command(gripper_action)
        
        # B. 手臂
        self.bridge.publish_target(self.nextpos)

        # 4. 延时 (控制频率)
        time.sleep(1.0 / self.hz) # 比如 10Hz = 0.1s

        # 5. 获取新状态
        self._update_currpos()
        obs = self._get_obs()
        
        # 计算 Reward (同 FrankaEnv)
        reward = self.compute_reward(obs)
        done = False # 根据需求定义
        return obs, int(reward), done, False, {}

    def _send_gripper_command(self, action_val):
        # 简单的二值化逻辑
        if action_val < -0.5: # 关
            self.bridge.move_gripper(0.8) # 0.8 是 Robotiq 闭合的大概位置
        elif action_val > 0.5: # 开
            self.bridge.move_gripper(0.0)

    def _update_currpos(self):
        # 1. 读位姿
        self.currpos = self.bridge.get_current_pose()
        
        # 2. 读速度 (需要自己算，或者读 joint velocity)
        # 简化：暂时设为 0，因为 Impedance Control 主要看位置
        self.currvel = np.zeros(6) 
        
        # 3. 读力/力矩 (⚠️ 重点：此处填0，避免维度报错)
        # 因为 Kinova 没有直接的 Cartesian Wrench 反馈给 Python (除非你写 Subscriber)
        self.currforce = np.zeros(3)
        self.currtorque = np.zeros(3)
        
        # 4. 读夹爪位置 (这里简化，假设指令即位置)
        self.curr_gripper_pos = np.array([0.0]) 

    def clip_safety_box(self, pose):
        # ⚠️ 必须修改这里的硬编码参数！
        # 拿到真机后，手动把机器推到桌子边缘，记录 XYZ
        pose[:3] = np.clip(pose[:3], 
                           [0.2, -0.3, 0.05], # Min X, Y, Z (Z至少要 > 0)
                           [0.6,  0.3, 0.5])  # Max X, Y, Z
        return pose
        
    def close(self):
        self.bridge.destroy_node()
        rclpy.shutdown()
        # 关闭相机
        for cap in self.cap.values():
            cap.close()