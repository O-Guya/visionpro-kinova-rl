## 安装环境

```bash
sudo apt-get install ros-noetic-trac-ik-python
conda create -n kinova_gen2 python==3.8 && conda activate kinova_gen2
pip install uv
uv pip compile requirements.in -o requirements.txt
uv pip sync requirements.txt

# 如果要添加 pip 包，则像下面这样运行后
uv pip install numpy
# 再把包名手动添加到 requirements.in，再更新目录
uv pip compile requirements.in -o requirements.txt
uv pip sync requirements.txt
```

[Kinect 安装参考](https://developer.aliyun.com/article/1592457#comment)，启动指令 `k4aviewer`

`conda` 环境下用 `kinect` 需要额外操作：
```bash
# 确认存在
ls /usr/lib/x86_64-linux-gnu/libdepthengine.so.2.0
# 写入环境变量
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> ~/.bashrc
source ~/.bashrc
```

## 排查指令

```bash
ps -ef | grep robot_state_publisher
ps -fp 184167   # 查看它的父节点
# 下述需要先手动打开驱动
rosservice call /j2s7s300_driver/in/start_force_control # 阻抗模式
rosservice call /j2s7s300_driver/in/home_arm # 出厂位置
rosservice call /j2s7s300_driver/in/start   # 恢复接收指令
rosservice call /j2s7s300_driver/in/stop    # 停止接收指令
rostopic pub -r 100 /j2s7s300_driver/in/joint_velocity_with_finger_velocity kinova_msgs/JointVelocityWithFingerVelocity "{joint1: 0.0, joint2: 0.0, joint3: 0.0, joint4: 0.0, joint5: 0.0, joint6: 0.0, joint7: 0.0, finger1: 4000.0, finger2: 4000.0, finger3: 4000.0}" # 速度控制同时控制关节和夹爪
```

## 启动机械臂

```bash
conda activate kinova_gen2
source ./devel/setup.bash

# 通过 MoveIt!+rviz 来控制机械臂
roslaunch kinova_teleop GUI_ctl.launch

# Vision Pro 调参模式
rosrun kinova_teleop avp_sim_mover.py tuner
rqt_plot /tuner/x/raw /tuner/x/filtered
rosrun kinova_teleop avp_real_mover.py tuner
rqt_plot /debug/joint/target/data[0] /debug/joint/current/data[0]

# 通过 VisonPro+MoveIt! 实现在仿真环境中遥操作
roslaunch kinova_teleop sim_robot_avp.launch
# 另开一个终端
cd ~/kinova_gen2_ws && conda activate kinova_gen2 && source ./devel/setup.bash
rosrun kinova_teleop avp_sim_mover.py

# 通过 VisonPro 实现真机遥操作
roslaunch kinova_teleop real_robot_avp.launch
# or headless
roslaunch kinova_teleop real_robot_avp.launch use_rviz:=false
# 另开一个终端
cd ~/kinova_gen2_ws && conda activate kinova_gen2 && source ./devel/setup.bash
rosrun kinova_teleop avp_real_mover.py
```
## 数采

```bash
cd ~/kinova_gen2_ws && conda activate kinova_gen2 && source ./devel/setup.bash
# 查找相机对应的 ID_PATH
python src/kinova_teleop/scripts/utils/find_camera_id.py 
# 启动机械臂驱动
roslaunch kinova_teleop real_robot_avp.launch
# 另开一个终端
cd ~/kinova_gen2_ws && conda activate kinova_gen2 && source ./devel/setup.bash
# 启动数采
rosrun kinova_teleop data_recorder.py --save_dir test
```
