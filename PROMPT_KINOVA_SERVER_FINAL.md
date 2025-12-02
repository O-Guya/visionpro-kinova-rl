# Prompt: HIL-SERL Kinova 服务端实现

## 任务目标

为 Kinova Gen3 编写 `kinova_server.py`，通过 Flask HTTP 接收 RL 策略指令，在底层实现导纳控制模拟柔顺性。

## 架构设计

**双线程异步**：
- **Flask 线程**：接收 HTTP 请求，更新全局变量 `target_action`
- **控制线程 (50Hz)**：读取状态 → 导纳修正 → 差分IK → 发送轨迹
- **Watchdog**：0.5秒无新指令触发减速停止

## 核心实现

### 1. 雅可比矩阵计算（KDL）

```python
from kdl_parser_py.urdf import treeFromUrdfModel
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
import numpy as np

class KinovaController:
    def __init__(self, urdf_path):
        # 构建KDL链
        robot = URDF.from_xml_file(urdf_path)
        ok, tree = treeFromUrdfModel(robot)
        self.chain = tree.getChain("base_link", "end_effector_link")

        # 雅可比求解器
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)

        # FK求解器（用于获取末端姿态）
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.chain)

    def get_jacobian(self, q: np.ndarray) -> np.ndarray:
        """计算雅可比矩阵 (6×7)"""
        q_kdl = kdl.JntArray(7)
        for i in range(7):
            q_kdl[i] = q[i]

        jac = kdl.Jacobian(7)
        self.jac_solver.JntToJac(q_kdl, jac)

        # 转换为numpy (6×7)
        J = np.zeros((6, 7))
        for i in range(6):
            for j in range(7):
                J[i, j] = jac[i, j]
        return J

    def get_ee_rotation(self, q: np.ndarray) -> np.ndarray:
        """获取末端旋转矩阵 (3×3)"""
        q_kdl = kdl.JntArray(7)
        for i in range(7):
            q_kdl[i] = q[i]

        frame = kdl.Frame()
        self.fk_solver.JntToCart(q_kdl, frame)

        R = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                R[i, j] = frame.M[i, j]
        return R
```

### 2. 导纳控制（含阻尼）

```python
class AdmittanceController:
    def __init__(self):
        # 参数（SI单位）
        self.K_admittance = np.diag([0.002]*3 + [0.0005]*3)  # [m/s/N, rad/s/Nm]
        self.D_damping = np.diag([10]*3 + [2]*3)             # [N·s/m, Nm·s/rad]
        self.force_deadband = 2.0  # N

        # 状态
        self.v_compliant = np.zeros(6)
        self.force_bias = np.zeros(6)  # 重力补偿

    def tare_force(self, force_current):
        """零点校准（在reset时调用）"""
        self.force_bias = force_current.copy()

    def compute_compliant_velocity(self, force_tool: np.ndarray, dt: float) -> np.ndarray:
        """
        计算导纳修正速度（一阶阻尼系统）

        v_compliant = K·F - D·v  （离散积分）
        """
        # 去除零点偏置
        force_tool = force_tool - self.force_bias

        # 死区滤波
        force_filtered = np.where(
            np.abs(force_tool) > self.force_deadband,
            force_tool,
            0.0
        )

        # 一阶导纳 + 阻尼
        dv = self.K_admittance @ force_filtered - self.D_damping @ self.v_compliant
        self.v_compliant += dv * dt

        # 限幅
        self.v_compliant = np.clip(self.v_compliant, -0.1, 0.1)

        return self.v_compliant
```

### 3. 控制循环

```python
import threading
import time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class KinovaServer:
    def __init__(self, urdf_path):
        self.controller = KinovaController(urdf_path)
        self.admittance = AdmittanceController()

        # 共享变量（需要锁）
        self.lock = threading.Lock()
        self.target_action = np.zeros(6)  # [dx,dy,dz,drx,dry,drz]
        self.last_update_time = time.time()
        self.running = True

        # 启动控制线程
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

    def _control_loop(self):
        """50Hz控制循环"""
        rate = Rate(50)  # ROS2 rate
        dt = 0.02

        while self.running:
            # 1. 读取状态
            q_current = self.get_joint_positions()  # 从ROS2订阅
            dq_current = self.get_joint_velocities()
            wrench_base = self.get_wrench_sensor()  # BaseCyclic

            # 2. 力坐标系转换（基座→末端）
            R_base_to_ee = self.controller.get_ee_rotation(q_current)
            force_tool = R_base_to_ee.T @ wrench_base[:3]
            torque_tool = R_base_to_ee.T @ wrench_base[3:]
            wrench_tool = np.concatenate([force_tool, torque_tool])

            # 3. 获取目标动作（线程安全）
            with self.lock:
                v_action = self.target_action.copy()
                time_since_update = time.time() - self.last_update_time

            # 4. Watchdog检查
            if time_since_update > 0.5:
                self._emergency_stop(q_current)
                rate.sleep()
                continue

            # 5. 导纳修正
            v_compliant = self.admittance.compute_compliant_velocity(wrench_tool, dt)
            v_total = v_action + v_compliant

            # 6. 差分IK
            J = self.controller.get_jacobian(q_current)
            J_pinv = np.linalg.pinv(J, rcond=0.01)  # 阻尼伪逆
            dq_cmd = J_pinv @ v_total

            # 7. 速度限幅
            dq_max = np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5])  # rad/s
            dq_cmd = np.clip(dq_cmd, -dq_max, dq_max)

            # 8. 积分到位置
            q_next = q_current + dq_cmd * dt

            # 9. 发送轨迹
            traj = JointTrajectory()
            traj.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4',
                                'joint_5', 'joint_6', 'joint_7']

            point = JointTrajectoryPoint()
            point.positions = q_next.tolist()
            point.velocities = dq_cmd.tolist()  # 关键：显式速度
            point.time_from_start = Duration(sec=0, nanosec=20_000_000)  # 20ms
            traj.points = [point]

            self.trajectory_pub.publish(traj)

            rate.sleep()

    def _emergency_stop(self, q_current):
        """软急停：0.5秒减速到零"""
        traj = JointTrajectory()
        traj.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4',
                            'joint_5', 'joint_6', 'joint_7']

        point = JointTrajectoryPoint()
        point.positions = q_current.tolist()
        point.velocities = [0.0] * 7
        point.time_from_start = Duration(sec=0, nanosec=500_000_000)
        traj.points = [point]

        self.trajectory_pub.publish(traj)
```

### 4. Flask接口

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
server = None  # 全局KinovaServer实例

@app.route('/update_pose', methods=['POST'])
def update_pose():
    """接收笛卡尔速度指令 [vx,vy,vz,wx,wy,wz]"""
    action = np.array(request.json['action'], dtype=np.float32)

    with server.lock:
        server.target_action = action
        server.last_update_time = time.time()

    return jsonify({"status": "ok"})

@app.route('/update_gripper', methods=['POST'])
def update_gripper():
    """夹爪控制 [0,1]"""
    gripper_pos = float(request.json['gripper'])
    server.send_gripper(gripper_pos)
    return jsonify({"status": "ok"})

@app.route('/get_state', methods=['POST'])
def get_state():
    """返回状态字典"""
    state = {
        "tcp_pose": server.get_ee_pose().tolist(),      # [x,y,z,qx,qy,qz,qw]
        "tcp_vel": server.get_ee_velocity().tolist(),   # [vx,vy,vz,wx,wy,wz]
        "tcp_force": server.get_force_tool()[:3].tolist(),
        "tcp_torque": server.get_force_tool()[3:].tolist(),
        "joint_positions": server.get_joint_positions().tolist(),
        "joint_velocities": server.get_joint_velocities().tolist(),
        "gripper_pose": [server.get_gripper_position()],
        "timestamp": time.time()
    }
    return jsonify(state)

@app.route('/reset', methods=['POST'])
def reset():
    """重置并校准力传感器"""
    server.reset_to_home()
    wrench_current = server.get_wrench_sensor()
    R = server.controller.get_ee_rotation(server.get_joint_positions())
    force_tool = R.T @ wrench_current[:3]
    torque_tool = R.T @ wrench_current[3:]
    server.admittance.tare_force(np.concatenate([force_tool, torque_tool]))
    return jsonify({"status": "reset_done"})

if __name__ == '__main__':
    import rclpy
    rclpy.init()

    server = KinovaServer(urdf_path="path/to/gen3.urdf")
    app.run(host='0.0.0.0', port=5000, threaded=True)
```

## 关键参数

```python
# 导纳控制
K_admittance = [0.002, 0.002, 0.002, 0.0005, 0.0005, 0.0005]  # 起点值
D_damping = [10, 10, 10, 2, 2, 2]                             # 阻尼
force_deadband = 2.0  # N

# 速度限制
dq_max = [1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5]  # rad/s（Kinova Gen3规格）

# 控制频率
control_hz = 50  # 最低要求，导纳控制需要
```

## 调试流程

1. **验证雅可比**：手动移动机器人，检查 `J @ dq` 是否匹配末端速度
2. **测试导纳**：推机器人应该后退，不震荡
3. **调整参数**：
   - 如果响应慢 → 增大 `K_admittance`
   - 如果震荡 → 增大 `D_damping`
   - 如果噪声大 → 增大 `force_deadband`

## 注意事项

- **URDF路径**：确保包含完整的DH参数
- **力传感器话题**：验证 `/my_gen3/base_feedback` 的wrench字段
- **JTC验证**：测试高频覆盖是否稳定（可能需要降到20Hz）
- **安全第一**：先在低 `K_admittance` 下测试，逐步增大
