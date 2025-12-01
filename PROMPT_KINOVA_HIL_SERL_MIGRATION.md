# Prompt: 迁移 HIL-SERL 到 Kinova Gen3 机器人

## 项目背景

当前有一个基于 Franka 机器人的 HIL-SERL 强化学习框架，需要迁移到 Kinova Gen3 7-DOF 机械臂。

## 硬件约束

**关键差异**：
1. **控制器类型**：
   - Franka：支持 `twist_controller`（笛卡尔速度控制）
   - **Kinova：仅支持 `joint_trajectory_controller`（关节轨迹控制）**

2. **相机配置**：
   - Franka：多个腕部相机
   - **Kinova：2个相机（wrist_cam + side_cam）**

3. **通信方式**：
   - Franka：通过 HTTP REST API
   - **Kinova：通过 ROS2 Humble + kortex_bringup**

## 现有资源

### 1. HIL-SERL 原始代码结构

```python
# hil-serl/serl_robot_infra/franka_env/envs/franka_env.py
class FrankaEnv(gym.Env):
    def __init__(self, hz=10, config=None):
        self.url = config.SERVER_URL  # HTTP控制接口
        self.action_space = gym.spaces.Box(-1, 1, shape=(7,))  # [dx,dy,dz,drx,dry,drz,gripper]
        self.observation_space = gym.spaces.Dict({
            "state": {
                "tcp_pose": (7,),      # [x,y,z,qx,qy,qz,qw]
                "tcp_vel": (6,),       # [vx,vy,vz,wx,wy,wz]
                "gripper_pose": (1,),
                "tcp_force": (3,),
                "tcp_torque": (3,),
            },
            "images": {
                "wrist_1": (128,128,3),
                "wrist_2": (128,128,3),
            }
        })

    def step(self, action):
        # action: [dx, dy, dz, drx, dry, drz, gripper]
        # 通过 twist_controller 直接发送笛卡尔增量
        requests.post(f"{self.url}/update_pose", json=action[:6])
        requests.post(f"{self.url}/update_gripper", json=action[6])
```

### 2. 已实现的 Kinova 遥操作模块

位于 `kinova_teleoperation/kinova_teleoperation/modules/`：

```python
# robot_interface.py - ROS2 通信封装
class KinovaRobotInterface:
    def send_trajectory(self, trajectory: JointTrajectory) -> bool
    def send_gripper(self, position: float) -> bool
    def get_state(self) -> Dict  # {joints, velocities, ee_pose, timestamp, gripper_position}
    def hold_position(self) -> bool

# motion_planner.py - IK求解器
class MotionPlanner:
    def solve_ik(
        self,
        target_position: np.ndarray,  # [x,y,z]
        target_rotation: np.ndarray,  # 3x3旋转矩阵
        seed_joints: np.ndarray       # 当前关节角（温启动）
    ) -> Optional[np.ndarray]  # 返回7个关节角或None

    def forward_kinematics(self, joint_angles: np.ndarray) -> (position, rotation)

    def generate_trajectory_window(
        self,
        target_joint_positions: np.ndarray,
        hand_velocity: np.ndarray,
        current_joints: np.ndarray
    ) -> JointTrajectory

# safety_monitor.py - 安全监控
class SafetyMonitor:
    def check_system_health(...) -> bool
    def clamp_to_workspace(target_position) -> clamped_position

# data_logger.py - 数据记录（已兼容HIL-SERL的HDF5格式）
class DataLogger:
    def log_frame(robot_state, action_delta, image)
```

## 任务要求

创建 **完整的 Kinova HIL-SERL 集成包**，包括：

### 核心任务

1. **创建 `KinovaEnv` Gymnasium 环境**
   - 继承 `gym.Env`
   - 实现标准接口：`reset()`, `step()`, `_get_obs()`
   - 动作空间：与 FrankaEnv 一致（7D 笛卡尔增量 + 夹爪）
   - 观察空间：适配2个相机，保持state字段兼容

2. **实现笛卡尔增量到关节轨迹的转换**
   - 输入：`action = [dx, dy, dz, drx, dry, drz, gripper]`（笛卡尔增量）
   - 过程：
     1. 获取当前末端位姿（通过FK或ROS2订阅）
     2. 计算目标位姿：`target_pose = current_pose + action[:6] * scale`
     3. 求解IK：`target_joints = IK(target_pose, seed=current_joints)`
     4. 生成轨迹：`JointTrajectory` 消息
     5. 发送到 `joint_trajectory_controller`
   - 关键：使用已有的 `MotionPlanner.solve_ik()` 和 `KinovaRobotInterface.send_trajectory()`

3. **集成相机接口**
   - 适配2个RealSense相机（wrist_cam, side_cam）
   - 参考 `hil-serl/franka_env/camera/rs_capture.py`
   - 图像尺寸：128×128×3（与HIL-SERL一致）
   - 实时捕获（20Hz或更高）

4. **实现安全机制**
   - 集成 `SafetyMonitor`
   - 工作空间限制（防止碰撞）
   - IK失败处理（保持当前位置）
   - 紧急停止机制

5. **配置管理**
   - 创建 `KinovaEnvConfig` 类（参考 `DefaultEnvConfig`）
   - 支持YAML配置文件
   - 参数：相机序列号、工作空间限制、控制频率等

6. **数据记录适配**
   - 确保观察/动作格式与HIL-SERL训练脚本兼容
   - 支持离线演示数据加载
   - 可选：集成已有的 `DataLogger`

## 详细实现规范

### 1. 动作空间转换（关键！）

```python
def step(self, action: np.ndarray):
    """
    action: [dx, dy, dz, drx, dry, drz, gripper]
    单位：dx,dy,dz in meters, drx,dry,drz in radians, gripper in [0,1]
    """
    # === 步骤1: 获取当前状态 ===
    robot_state = self.robot_interface.get_state()
    current_joints = robot_state['joints']  # (7,)
    current_ee_pose = robot_state['ee_pose']  # [x,y,z,qx,qy,qz,qw]

    # 提取当前位置和旋转
    current_position = current_ee_pose[:3]  # [x,y,z]
    current_quat = current_ee_pose[3:]      # [qx,qy,qz,qw]
    current_rotation = Rotation.from_quat(current_quat).as_matrix()  # 3x3

    # === 步骤2: 计算目标位姿 ===
    # 笛卡尔增量（需要缩放）
    delta_position = action[:3] * self.action_scale  # 例如 action_scale=0.01 (1cm)
    delta_rotation_vec = action[3:6] * self.rotation_scale  # 例如 rotation_scale=0.1 (rad)

    target_position = current_position + delta_position

    # 旋转增量（轴角表示）
    delta_rotation = Rotation.from_rotvec(delta_rotation_vec)
    target_rotation = delta_rotation.as_matrix() @ current_rotation

    # === 步骤3: 安全检查 ===
    target_position, _ = self.safety_monitor.clamp_to_workspace(target_position)

    # === 步骤4: 求解IK ===
    target_joints = self.motion_planner.solve_ik(
        target_position=target_position,
        target_rotation=target_rotation,
        seed_joints=current_joints  # 温启动
    )

    if target_joints is None:
        # IK失败，保持当前位置
        self.robot_interface.hold_position()
        print("[WARNING] IK failed, holding position")
        return self._get_obs(), 0.0, False, False, {"ik_failed": True}

    # === 步骤5: 生成轨迹并发送 ===
    trajectory = self._create_trajectory(
        target_joints=target_joints,
        current_joints=current_joints,
        duration=1.0/self.hz  # 例如 0.1s for 10Hz
    )

    self.robot_interface.send_trajectory(trajectory)

    # === 步骤6: 夹爪控制 ===
    gripper_position = action[6]  # [-1, 1] -> [0, 1]
    gripper_normalized = (gripper_position + 1.0) / 2.0
    self.robot_interface.send_gripper(gripper_normalized)

    # === 步骤7: 等待执行 ===
    time.sleep(1.0 / self.hz)

    # === 步骤8: 获取观察和奖励 ===
    obs = self._get_obs()
    reward = self._compute_reward(obs)
    done = self._check_done(obs)

    return obs, reward, done, False, {}

def _create_trajectory(self, target_joints, current_joints, duration):
    """创建单点轨迹"""
    trajectory = JointTrajectory()
    trajectory.joint_names = self.robot_interface.joint_names

    point = JointTrajectoryPoint()
    point.positions = target_joints.tolist()

    # 计算速度（有限差分）
    velocities = (target_joints - current_joints) / duration
    point.velocities = velocities.tolist()

    point.time_from_start = Duration(
        sec=0,
        nanosec=int(duration * 1e9)
    )

    trajectory.points.append(point)
    return trajectory
```

### 2. 观察空间定义

```python
def _get_obs(self) -> Dict:
    """获取观察"""
    # 机器人状态
    robot_state = self.robot_interface.get_state()

    # 相机图像
    images = {}
    for cam_name, cam_obj in self.cameras.items():
        images[cam_name] = cam_obj.read()  # (H, W, 3)
        images[cam_name] = cv2.resize(images[cam_name], (128, 128))

    obs = {
        "state": {
            "tcp_pose": robot_state['ee_pose'].astype(np.float32),  # (7,)
            "tcp_vel": self._compute_tcp_velocity(),  # (6,) 需要实现
            "gripper_pose": np.array([robot_state['gripper_position']], dtype=np.float32),

            # Kinova Gen3 有力传感器，需通过ROS2获取
            "tcp_force": self._get_force_sensor(),    # (3,) 如果没有，返回zeros
            "tcp_torque": self._get_torque_sensor(),  # (3,) 如果没有，返回zeros
        },
        "images": images  # {"wrist_cam": (128,128,3), "side_cam": (128,128,3)}
    }

    return obs
```

### 3. 相机接口

```python
class KinovaCameraInterface:
    """管理多个RealSense相机"""

    def __init__(self, camera_serials: Dict[str, str]):
        """
        camera_serials: {"wrist_cam": "123456", "side_cam": "789012"}
        """
        self.cameras = {}

        for name, serial in camera_serials.items():
            self.cameras[name] = RSCapture(
                serial_number=serial,
                width=640,
                height=480,
                fps=30
            )

    def get_images(self) -> Dict[str, np.ndarray]:
        """获取所有相机图像"""
        images = {}
        for name, cam in self.cameras.items():
            rgb, depth = cam.read()
            # 裁剪和缩放到128x128
            images[name] = cv2.resize(rgb, (128, 128))
        return images

    def close(self):
        """关闭所有相机"""
        for cam in self.cameras.values():
            cam.close()
```

### 4. 配置类

```python
class KinovaEnvConfig:
    """Kinova环境配置"""

    # 机器人参数
    ROBOT_NAME: str = "my_gen3"
    URDF_PATH: str = "../ros2_kortex/kortex_description/arms/gen3/7dof/urdf/gen3_macro.xacro"

    # 相机配置
    REALSENSE_CAMERAS: Dict[str, str] = {
        "wrist_cam": "YOUR_WRIST_CAM_SERIAL",
        "side_cam": "YOUR_SIDE_CAM_SERIAL",
    }

    # 控制参数
    HZ: int = 10  # 控制频率（HIL-SERL默认10Hz）
    ACTION_SCALE: float = 0.01  # 位置增量缩放（1cm per action unit）
    ROTATION_SCALE: float = 0.1  # 旋转增量缩放（0.1 rad per action unit）

    # 工作空间限制（从safety_params.yaml加载）
    SAFETY_CONFIG_PATH: str = "config/safety_params.yaml"

    # 重置位置（关节空间，7个关节角度）
    RESET_JOINTS: np.ndarray = np.array([0.0, 0.5, 0.0, 2.0, 0.0, 1.0, 0.0])

    # 重置位置（笛卡尔空间，[x,y,z,rx,ry,rz]）
    RESET_POSE: np.ndarray = np.array([0.3, 0.0, 0.4, 3.14, 0.0, 0.0])

    # 奖励计算（任务相关）
    TARGET_POSE: np.ndarray = np.array([0.4, 0.2, 0.3, 3.14, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])

    # 其他
    MAX_EPISODE_LENGTH: int = 200
    DISPLAY_IMAGE: bool = True
```

### 5. Reset 实现

```python
def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
    """重置环境到初始状态"""
    super().reset(seed=seed)

    # 方案1: 移动到预定义的关节位置（推荐，更可靠）
    reset_trajectory = self._create_reset_trajectory(
        target_joints=self.config.RESET_JOINTS,
        duration=3.0  # 3秒缓慢移动到初始位置
    )
    self.robot_interface.send_trajectory(reset_trajectory)
    time.sleep(3.5)  # 等待到达

    # 方案2: 移动到笛卡尔位置（需要IK）
    # reset_joints = self.motion_planner.solve_ik(
    #     target_position=self.config.RESET_POSE[:3],
    #     target_rotation=Rotation.from_euler('xyz', self.config.RESET_POSE[3:]).as_matrix(),
    #     seed_joints=self.robot_interface.get_state()['joints']
    # )

    # 重置夹爪（打开）
    self.robot_interface.send_gripper(0.0)
    time.sleep(1.0)

    # 重置计数器
    self.episode_step = 0

    # 返回初始观察
    obs = self._get_obs()
    info = {}

    return obs, info

def _create_reset_trajectory(self, target_joints, duration):
    """创建重置轨迹"""
    trajectory = JointTrajectory()
    trajectory.joint_names = self.robot_interface.joint_names

    point = JointTrajectoryPoint()
    point.positions = target_joints.tolist()
    point.velocities = [0.0] * 7  # 最终速度为0
    point.time_from_start = Duration(sec=int(duration), nanosec=0)

    trajectory.points.append(point)
    return trajectory
```

### 6. 奖励函数（任务相关，示例）

```python
def _compute_reward(self, obs: Dict) -> float:
    """计算奖励（示例：reaching任务）"""
    current_pose = obs['state']['tcp_pose'][:3]  # [x,y,z]
    target_pose = self.config.TARGET_POSE[:3]

    # 距离奖励
    distance = np.linalg.norm(current_pose - target_pose)
    reward = -distance

    # 如果到达目标（阈值内），给予额外奖励
    if distance < 0.02:  # 2cm
        reward += 10.0

    return reward

def _check_done(self, obs: Dict) -> bool:
    """检查episode是否结束"""
    # 超过最大步数
    if self.episode_step >= self.config.MAX_EPISODE_LENGTH:
        return True

    # 到达目标
    current_pose = obs['state']['tcp_pose'][:3]
    target_pose = self.config.TARGET_POSE[:3]
    if np.linalg.norm(current_pose - target_pose) < 0.02:
        return True

    # 安全违规（超出工作空间）
    if not self.safety_monitor.is_position_safe(current_pose):
        return True

    return False
```

## 项目结构

```
kinova_hil_serl/
├── kinova_hil_serl/
│   ├── __init__.py
│   ├── kinova_env.py              # 主环境类 KinovaEnv
│   ├── kinova_config.py           # KinovaEnvConfig配置类
│   ├── camera_interface.py        # 相机接口封装
│   └── utils/
│       ├── __init__.py
│       ├── ik_solver.py           # IK求解器包装（基于MotionPlanner）
│       └── trajectory_utils.py    # 轨迹生成工具
├── config/
│   ├── kinova_env_config.yaml     # 环境配置
│   └── camera_serials.yaml        # 相机序列号
├── scripts/
│   ├── test_env.py                # 测试环境（随机动作）
│   ├── test_cameras.py            # 测试相机连接
│   └── calibrate_workspace.py     # 工作空间校准
├── examples/
│   ├── collect_demo.py            # 遥操作收集演示
│   ├── replay_demo.py             # 回放演示验证
│   └── train_policy.py            # HIL-SERL训练脚本（适配版）
├── setup.py
├── requirements.txt
└── README.md
```

## 关键技术点

### 1. IK求解器性能优化

```python
# 使用温启动（warm start）提高成功率和速度
def solve_ik_with_fallback(self, target_position, target_rotation, current_joints):
    """IK求解器带降级方案"""

    # 尝试1: 使用当前关节角作为种子
    result = self.motion_planner.solve_ik(
        target_position, target_rotation, seed_joints=current_joints
    )
    if result is not None:
        return result

    # 尝试2: 使用预定义的多个种子位置
    for seed in self.ik_seeds:
        result = self.motion_planner.solve_ik(
            target_position, target_rotation, seed_joints=seed
        )
        if result is not None:
            return result

    # 尝试3: 放宽位置容限
    self.motion_planner.position_tolerance = 0.005  # 5mm
    result = self.motion_planner.solve_ik(
        target_position, target_rotation, seed_joints=current_joints
    )
    self.motion_planner.position_tolerance = 0.001  # 恢复

    return result  # 可能仍为None
```

### 2. 速度计算（用于tcp_vel）

```python
def _compute_tcp_velocity(self) -> np.ndarray:
    """计算末端速度"""
    current_state = self.robot_interface.get_state()
    current_ee_pose = current_state['ee_pose']
    current_time = time.time()

    if self.last_ee_pose is None:
        self.last_ee_pose = current_ee_pose
        self.last_time = current_time
        return np.zeros(6)

    # 位置速度
    dt = current_time - self.last_time
    if dt < 1e-6:
        return np.zeros(6)

    linear_vel = (current_ee_pose[:3] - self.last_ee_pose[:3]) / dt

    # 角速度（四元数差分）
    q_current = current_ee_pose[3:]
    q_last = self.last_ee_pose[3:]

    # 四元数差分到角速度
    q_diff = self._quat_multiply(q_current, self._quat_conjugate(q_last))
    angular_vel = 2.0 * q_diff[1:] / dt  # 近似

    # 更新历史
    self.last_ee_pose = current_ee_pose
    self.last_time = current_time

    return np.concatenate([linear_vel, angular_vel])
```

### 3. 力/力矩传感器接口

```python
class ForceTorqueSensor:
    """Kinova Gen3 力传感器接口"""

    def __init__(self, robot_name: str):
        self.node = rclpy.create_node('force_sensor_reader')
        self.sub = self.node.create_subscription(
            WrenchStamped,
            f'/{robot_name}/force_torque_sensor',
            self._callback,
            10
        )
        self.latest_wrench = None

    def _callback(self, msg: WrenchStamped):
        self.latest_wrench = msg

    def get_force(self) -> np.ndarray:
        if self.latest_wrench is None:
            return np.zeros(3)
        return np.array([
            self.latest_wrench.wrench.force.x,
            self.latest_wrench.wrench.force.y,
            self.latest_wrench.wrench.force.z
        ])

    def get_torque(self) -> np.ndarray:
        if self.latest_wrench is None:
            return np.zeros(3)
        return np.array([
            self.latest_wrench.wrench.torque.x,
            self.latest_wrench.wrench.torque.y,
            self.latest_wrench.wrench.torque.z
        ])
```

### 4. 测试脚本

```python
# scripts/test_env.py
def test_kinova_env():
    """测试环境基本功能"""
    import gymnasium as gym
    from kinova_hil_serl import KinovaEnv, KinovaEnvConfig

    config = KinovaEnvConfig()
    env = KinovaEnv(config=config)

    print("Testing reset...")
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"State shape: {obs['state']['tcp_pose'].shape}")
    print(f"Image keys: {obs['images'].keys()}")

    print("\nTesting random actions...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, done={done}")

        if done:
            obs, info = env.reset()

    env.close()
    print("\nTest completed!")

if __name__ == "__main__":
    test_kinova_env()
```

## 集成步骤

### Step 1: 创建基础环境（Day 1）
```bash
1. 创建 kinova_hil_serl 包结构
2. 实现 KinovaEnv 基本框架（reset, step, _get_obs）
3. 集成 RobotInterface 和 MotionPlanner
4. 测试 IK 转换流程
```

### Step 2: 集成相机（Day 2）
```bash
1. 实现 KinovaCameraInterface
2. 识别相机序列号（rs-enumerate-devices）
3. 测试图像采集
4. 集成到观察空间
```

### Step 3: 测试和调优（Day 3）
```bash
1. 运行 test_env.py 验证基本功能
2. 调整 action_scale 和 rotation_scale
3. 测试工作空间限制
4. 性能优化（IK成功率、控制延迟）
```

### Step 4: HIL-SERL 集成（Day 4）
```bash
1. 修改 HIL-SERL 训练脚本加载 KinovaEnv
2. 收集演示数据
3. 训练策略
4. 在线评估
```

## 性能要求

- **控制频率**：10Hz（与HIL-SERL一致）
- **IK成功率**：>90%（正常工作空间内）
- **端到端延迟**：<100ms（接收action → 执行完成）
- **相机帧率**：≥20Hz
- **Episode长度**：100-200步（10-20秒）

## 注意事项

1. **安全第一**：
   - 在真实机器人上测试前，先在仿真环境验证
   - 始终保持急停按钮在手边
   - 使用较小的 action_scale 开始（0.005m）

2. **IK失败处理**：
   - IK失败时，保持当前位置（不要尝试执行）
   - 记录失败率，如果>10%，说明动作范围过大

3. **关节限位**：
   - Kinova Gen3 各关节有限位，IK求解器会考虑
   - 如果频繁触及限位，调整工作空间范围

4. **相机同步**：
   - 两个相机需要硬件同步或软件对齐时间戳
   - 优先使用硬件触发同步

5. **ROS2线程**：
   - ROS2回调在独立线程，需要线程锁保护共享状态
   - 已有的 RobotInterface 已实现线程安全

## 可选增强功能

1. **轨迹平滑**：在 `generate_trajectory_window` 中添加多点轨迹
2. **碰撞检测**：集成 MoveIt2 场景规划
3. **视觉伺服**：使用相机进行闭环控制
4. **自适应IK**：根据成功率动态调整容限

## 交付标准

- [ ] `KinovaEnv` 类实现完整且通过测试
- [ ] 相机接口工作正常，能采集图像
- [ ] IK转换成功率 >90%
- [ ] 与HIL-SERL训练脚本集成成功
- [ ] 文档完善（README + API文档）
- [ ] 示例脚本可运行（test_env.py, collect_demo.py）

---

请基于以上规范，实现完整的 Kinova HIL-SERL 集成包。优先确保核心功能（IK转换、轨迹控制）的正确性，然后再优化性能和添加增强功能。
