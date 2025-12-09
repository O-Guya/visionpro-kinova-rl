#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import time
import sys
import yaml
import argparse
import os
import glob
from termios import tcgetattr, tcsetattr, TCSADRAIN
from tty import setcbreak
import select

from utils.robot_interface import RobotManager
from utils.camera import CameraManager
from utils.data_utils import UnifiedSaver


class KeyPoller:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_term = tcgetattr(self.fd)
        setcbreak(self.fd)
        return self

    def __exit__(self, type, value, traceback):
        tcsetattr(self.fd, TCSADRAIN, self.old_term)

    def poll(self):
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if dr else None


class DataRecorder:
    def __init__(self, config_path, save_dir=None):
        print(f"[DataRecorder] Loading config from: {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # 优先使用命令行参数，否则用 yaml 配置
        if save_dir:
            final_save_dir = f"data_collected/{save_dir}"
        else:
            final_save_dir = self.cfg.get("save_dir", "data_collected/test")
        save_format = self.cfg.get("save_format", "hdf5")

        # 初始化 RobotManager
        robot_configs = self.cfg.get("robots")
        if not robot_configs:
            print(
                "[DataRecorder] 'robots' key not found in yaml, using default AVP config."
            )
            robot_configs = [{"name": "main_arm", "type": "avp"}]

        self.robot_manager = RobotManager(robot_configs, use_vision=True)

        # 获取主要录制的机器人实例
        # 目前是单臂录制模式，默认取第一个机器人
        self.active_robot = self.robot_manager.get_robot()
        if self.active_robot is None:
            raise RuntimeError("No robot initialized via RobotManager!")

        print(f"[DataRecorder] Active Robot: {self.active_robot.name}")

        self.camera_manager = CameraManager(self.cfg.get("cameras", []))

        print(f"[DataRecorder] Saving to [{save_format.upper()}] in: {final_save_dir}")
        print(f"[DataRecorder] Naming convention: {{index}}.{save_format}")

        # 初始化 UnifiedSaver
        self.saver = UnifiedSaver(save_dir=final_save_dir, save_format=save_format)

        self.frequency = self.cfg.get("frequency", 30)
        self.rate = rospy.Rate(self.frequency)

    def _init_buffer(self):
        has_depth_cams = []
        for name, cam in self.camera_manager.cameras.items():
            if cam.use_depth:
                has_depth_cams.append(name)

        return {
            "state": [],
            "action": [],
            "state_gpos": [],
            "action_gpos": [],
            "velocity": [],
            "effort": [],
            "images": {name: [] for name in self.camera_manager.cameras.keys()},
            "depth": {name: [] for name in has_depth_cams},
        }

    def _scan_existing_episodes(self):
        """扫描目录下纯数字命名的文件 (例如 0.hdf5, 12.mcap)"""
        save_dir = self.saver.save_dir
        if not os.path.exists(save_dir):
            return 0

        # 扫描所有 .hdf5 和 .mcap
        files = glob.glob(os.path.join(save_dir, "*.hdf5")) + glob.glob(
            os.path.join(save_dir, "*.mcap")
        )

        max_idx = -1
        for f in files:
            try:
                basename = os.path.basename(f)
                name_without_ext = os.path.splitext(basename)[0]

                # [关键] 只有纯数字的文件名才会被计入
                if name_without_ext.isdigit():
                    idx = int(name_without_ext)
                    if idx > max_idx:
                        max_idx = idx
            except Exception:
                pass

        next_idx = max_idx + 1
        print(f"[DataRecorder] Resuming from index: {next_idx}")
        return next_idx

    def run(self):
        print("\n>>> Recorder Initialized (Sync Mode).")
        episode_idx = self._scan_existing_episodes()

        try:
            with KeyPoller() as key_poller:
                while not rospy.is_shutdown():
                    print(
                        f"\n[Episode {episode_idx}] Ready. SPACE=Start. 'r'=Delete Last."
                    )

                    # === 1. 待机 ===
                    while not rospy.is_shutdown():
                        key = key_poller.poll()
                        if key == " ":
                            break
                        elif key == "\x03":
                            return
                        elif key == "r":
                            # 删除上一条逻辑
                            if episode_idx > 0:
                                last_idx = episode_idx - 1
                                deleted = False
                                # 尝试删除 .hdf5 或 .mcap
                                for ext in [".hdf5", ".mcap"]:
                                    f_path = os.path.join(
                                        self.saver.save_dir, f"{last_idx}{ext}"
                                    )
                                    if os.path.exists(f_path):
                                        try:
                                            os.remove(f_path)
                                            print(f"🗑️ Deleted: {last_idx}{ext}")
                                            deleted = True
                                        except Exception as e:
                                            print(f"❌ Failed to delete {f_path}: {e}")

                                if deleted:
                                    episode_idx = last_idx
                                else:
                                    print(f"File {last_idx}.* not found.")
                        time.sleep(0.01)

                    if rospy.is_shutdown():
                        break

                    # === 2. 录制 ===
                    print(">>> 🔴 RECORDING... 's'=Save, 'q'=Discard")

                    # [修改 4] 使用 active_robot 进行控制
                    self.active_robot.enable()

                    self.raw_buffer = self._init_buffer()
                    is_running = True
                    save_flag = False

                    while is_running and not rospy.is_shutdown():
                        key = key_poller.poll()
                        if key == "s":
                            save_flag = True
                            is_running = False
                        elif key == "q":
                            is_running = False
                        elif key == "r":  # 录制中按 r 也是放弃
                            save_flag = False
                            is_running = False
                        elif key == "\x03":
                            return

                        # [修改 5] 数据采集调用 active_robot
                        qpos = self.active_robot.get_qpos()
                        action = self.active_robot.get_action()
                        qpos_gpos = self.active_robot.get_gpos_state()
                        action_gpos = self.active_robot.get_gpos_action()
                        velocity = self.active_robot.get_qvel()
                        effort = self.active_robot.get_qeffort()

                        rgb_dict, depth_dict = self.camera_manager.get_data()

                        self.raw_buffer["state"].append(qpos)
                        self.raw_buffer["action"].append(action)
                        self.raw_buffer["state_gpos"].append(qpos_gpos)
                        self.raw_buffer["action_gpos"].append(action_gpos)
                        self.raw_buffer["velocity"].append(velocity)
                        self.raw_buffer["effort"].append(effort)

                        for k, v in rgb_dict.items():
                            self.raw_buffer["images"][k].append(v)
                        for k, v in depth_dict.items():
                            if k in self.raw_buffer["depth"]:
                                self.raw_buffer["depth"][k].append(v)

                        self.rate.sleep()

                    # [修改 6] 复位机器人
                    self.active_robot.home()

                    # === 3. 保存 (串行阻塞) ===
                    if save_flag:
                        if len(self.raw_buffer["action"]) > 10:
                            print(f">>> 💾 Saving Episode {episode_idx} (Blocking)...")

                            # 构造符合 DataSaver 接口的字典
                            final_data = {
                                "action": self.raw_buffer["action"],
                                "action_gpos": self.raw_buffer["action_gpos"],
                                "observation": {
                                    "state": self.raw_buffer["state"],
                                    "state_gpos": self.raw_buffer["state_gpos"],
                                    "velocity": self.raw_buffer["velocity"],
                                    "effort": self.raw_buffer["effort"],
                                    "images": self.raw_buffer["images"],
                                    "depth": self.raw_buffer["depth"],
                                },
                            }

                            # 直接调用保存，主线程会在这里等待直到写入完成
                            success = self.saver.save_episode(final_data, episode_idx)

                            if success:
                                print(f">>> ✅ Episode {episode_idx} Saved.")
                                episode_idx += 1
                            else:
                                print(">>> ❌ Save Failed!")
                        else:
                            print(">>> Too short, discarded.")
                    else:
                        print(">>> Discarded.")

        except Exception as e:
            print(f"\n[Error] {e}")
            import traceback

            traceback.print_exc()
        finally:
            print("\n>>> Closing resources...")
            # 也可以在这里调用 self.robot_manager.home_all() 以防万一
            if hasattr(self, "active_robot") and self.active_robot:
                pass  # 已经在循环里 home 过了，这里不再重复，避免重复等待
            self.camera_manager.close()
            print(">>> Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="src/kinova_teleop/scripts/utils/config.yaml"
    )
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    rospy.init_node("data_recorder", anonymous=True)
    recorder = DataRecorder(config_path=args.config, save_dir=args.save_dir)
    recorder.run()
