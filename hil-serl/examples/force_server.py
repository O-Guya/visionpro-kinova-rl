import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
import pinocchio as pin
import numpy as np

class ForceObserver(Node):
    def __init__(self):
        super().__init__('force_observer')
        
        # === 1. 加载模型 ===
        # 指向我们刚才生成的临时文件
        urdf_path = "/tmp/gen3.urdf" 
        
        try:
            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()
            self.get_logger().info(f"成功加载模型: {urdf_path}")
        except Exception as e:
            self.get_logger().error(f"加载模型失败，请确认第1步操作已完成！错误: {e}")
            raise e

        # === 2. 定义正确的关节顺序 ===
        # 我们只关心这7个，必须按顺序排列
        self.target_joints = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4', 
            'joint_5', 'joint_6', 'joint_7'
        ]

        # 发布与订阅
        self.wrench_pub = self.create_publisher(WrenchStamped, '/estimated_wrench', 10)
        self.sub = self.create_subscription(JointState, '/joint_states', self.callback, 10)
        
        # 平滑滤波 (可选)
        self.wrench_filter = np.zeros(6)
        self.alpha = 0.2 # 滤波系数，越小越平滑但延迟越高

    def callback(self, msg):
        q = []
        tau = []
        
        # === 3. 关键逻辑：按名字提取数据 (解决乱序问题) ===
        try:
            for name in self.target_joints:
                if name not in msg.name:
                    return # 数据不全，跳过
                
                idx = msg.name.index(name)
                
                # 提取位置
                q.append(msg.position[idx])
                
                # 提取力矩
                eff = msg.effort[idx]
                # 你的夹爪 joint 是 .nan，但手臂 joint 应该是有数的
                # 以防万一，如果手臂也是 nan，这就没法算了
                if np.isnan(eff): 
                    # 只有刚启动的一瞬间可能是nan
                    return 
                tau.append(eff)
                
        except ValueError:
            return

        q = np.array(q)
        tau = np.array(tau)

        # === 4. 动力学计算 (Pinocchio) ===
        # 更新模型状态
        pin.computeJointJacobians(self.model, self.data, q)
        pin.framesForwardKinematics(self.model, self.data, q)
        
        # 获取末端 Frame ID (通常是最后一个 link)
        # 也可以按名字找: self.model.getFrameId("tool_frame")
        frame_id = self.model.nframes - 1 
        
        # 计算雅可比矩阵 (6x7)
        J = pin.getFrameJacobian(self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        # 计算重力项 (Gravity Compensation)
        g = pin.computeGeneralizedGravity(self.model, self.data, q)
        
        # 算出“外部力矩” = 传感器读数 - 重力影响
        tau_external = tau - g
        
        # 求解 F = (J^T)^+ * tau_ext
        # 使用伪逆矩阵 (Pseudo-Inverse)
        J_pinv = np.linalg.pinv(J.T)
        wrench = J_pinv @ tau_external

        # === 5. 简单滤波并发布 ===
        self.wrench_filter = (1 - self.alpha) * self.wrench_filter + self.alpha * wrench
        
        msg_out = WrenchStamped()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.header.frame_id = "base_link" # 力的参考系
        
        # 填入力
        msg_out.wrench.force.x = self.wrench_filter[0]
        msg_out.wrench.force.y = self.wrench_filter[1]
        msg_out.wrench.force.z = self.wrench_filter[2]
        # 填入力矩
        msg_out.wrench.torque.x = self.wrench_filter[3]
        msg_out.wrench.torque.y = self.wrench_filter[4]
        msg_out.wrench.torque.z = self.wrench_filter[5]
        
        self.wrench_pub.publish(msg_out)

def main():
    rclpy.init()
    node = ForceObserver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()