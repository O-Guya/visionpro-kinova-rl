#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
from abc import ABC, abstractmethod

from avp_real_mover import KinovaBaseMover, AVPTeleopMover


class AbstractRobot(ABC):
    def __init__(self, cfg):
        self.name = cfg.get("name", "robot")
        # 默认动作维度 (7关节 + 1夹爪)
        self.action_dim = cfg.get("action_dim", 8)

    @abstractmethod
    def enable(self):
        pass

    @abstractmethod
    def home(self):
        pass

    @abstractmethod
    def get_qpos(self):
        pass

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def get_gpos_state(self):
        pass

    @abstractmethod
    def get_gpos_action(self):
        pass

    @abstractmethod
    def get_qvel(self):
        pass

    @abstractmethod
    def get_qeffort(self):
        pass

    @abstractmethod
    def exec_action(self, action):
        pass


class Kinova_Gen2(AbstractRobot):
    """
    具体的 kinova_gen2 机械臂实现
    """

    def __init__(self, cfg, use_vision=False):
        super().__init__(cfg)

        self.use_vision = use_vision

        mode_str = (
            "[*] AVP Teleop (VR)" if self.use_vision else "[*] Base IK (Inference)"
        )
        print(f"[{self.name}] Initializing Mode: [{mode_str}]")

        if self.use_vision:
            if AVPTeleopMover is None:
                raise ImportError("AVPTeleopMover missing.")
            # 录制模式：加载遥操类
            self.mover = AVPTeleopMover(standalone=False)
        else:
            if KinovaBaseMover is None:
                raise ImportError("KinovaBaseMover missing.")
            # 推理模式：加载基础类
            self.mover = KinovaBaseMover()

    def enable(self):
        return self.mover.enable_control()

    def home(self):
        print(f"[{self.name}] Stopping control...")
        self.mover.disable_control()
        time.sleep(0.5)
        print(f"[{self.name}] Homing...")
        for i in range(3):
            try:
                self.mover.go_to_ready_pose()
                time.sleep(0.5)
                break
            except Exception as e:
                print(f"[Warn] Home attempt {i+1} failed: {e}")
                time.sleep(1.0)

    def get_qpos(self):
        return np.array(self.mover.get_current_joints(), dtype=np.float32)

    def get_action(self):
        return np.array(self.mover.get_target_joints(), dtype=np.float32)

    def get_gpos_state(self):
        return np.array(self.mover.get_current_cartesian(), dtype=np.float32)

    def get_gpos_action(self):
        return np.array(self.mover.get_target_cartesian(), dtype=np.float32)

    def get_qvel(self):
        with self.mover.lock:
            vels = list(self.mover.current_joint_velocities)
        return np.array(vels + [0.0], dtype=np.float32)

    def get_qeffort(self):
        with self.mover.lock:
            effs = list(self.mover.current_joint_efforts)
        return np.array(effs + [0.0], dtype=np.float32)

    def exec_action(self, action):
        if self.mover is None:
            return
        if len(action) < 8:
            return
        target_joints = action[:7]
        gripper_val = action[7]
        self.mover.set_target_joints(target_joints, gripper_val)


class RobotManager:
    """
    类似于 CameraManager，负责根据配置初始化具体的机器人
    """

    def __init__(self, robot_config_list, use_vision=False):
        self.robots = {}
        self.robot_list = []

        for r_cfg in robot_config_list:
            r_type = r_cfg.get("name", "kinova_gen2")

            if r_type == "kinova_gen2":
                bot = Kinova_Gen2(r_cfg, use_vision=use_vision)
            else:
                print(f"[RobotManager] Unknown robot type: {r_type}")
                continue

            self.robots[r_cfg["name"]] = bot
            self.robot_list.append(bot)
            print(f"[*] Robot {r_cfg['name']} initialized.")

    def enable_all(self):
        for bot in self.robot_list:
            bot.enable()

    def home_all(self):
        for bot in self.robot_list:
            bot.home()

    def get_robot(self, name=None):
        """
        获取指定名称的机器人，如果不传且只有一个，则返回第一个
        """
        if name is None:
            if len(self.robot_list) > 0:
                return self.robot_list[0]
            return None
        return self.robots.get(name)
