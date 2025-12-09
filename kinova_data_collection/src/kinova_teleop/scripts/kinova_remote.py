#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import numpy as np
import requests
from collections import deque
from PIL import Image
import io
import yaml
import argparse
import sys
import select
import cv2
import os
import threading
from termcolor import cprint
import termios
from tty import setcbreak

from utils.robot_interface import RobotManager
from utils.camera import CameraManager

# 用于控制终端的ANSI转义码
CURSOR_UP_ONE = "\x1b[1A"
ERASE_LINE = "\x1b[2K\r"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"


class KinovaRemoteController:
    """
    Kinova Gen2 机器人远程控制器 (Airbot 接口兼容版)
    """

    WINDOW_NAME = "Kinova Real-time Feeds"

    def __init__(self, config_path, server_ip, server_port):
        self.load_config(config_path)

        self.server_url = f"http://{server_ip}:{server_port}"
        self.max_steps = self.cfg.get("episode_length_limit", 1000)
        self.control_freq = self.cfg.get("frequency", 20)
        self.jpg_quality = 80

        # === 相机映射配置 ===
        # 键: 本地 config.yaml 里的名字
        # 值: 发送给服务器的名字 (训练时的名字)
        self.camera_mapping = {
            "cam_fixed": "cam_high",  # 示例: 本地叫 cam_fixed -> 服务器叫 cam_high
            "cam_wrist": "cam_right_wrist",  # 示例: 本地叫 cam_wrist -> 服务器叫 cam_right_wrist
            # "cam_left": "cam_left_wrist"
        }
        # 显示顺序
        self.display_order = ["cam_fixed", "cam_wrist"]

        self.session = requests.Session()

        # === 初始化硬件 ===
        cprint("[*] 初始化 RobotManager...", "cyan")
        # 兼容处理: 如果yaml里没有robots字段，手动构造
        robot_configs = self.cfg.get(
            "robots", [{"name": "kinova_gen2", "type": "kinova_gen2"}]
        )
        self.robot_manager = RobotManager(robot_configs)
        self.robot = self.robot_manager.get_robot()

        if self.robot is None:
            raise RuntimeError("无法初始化机器人，请检查 config.yaml")

        cprint("[*] 初始化 CameraManager...", "cyan")
        self.camera_manager = CameraManager(self.cfg.get("cameras", []))

        # === 线程与缓存 ===
        self.latest_observation_cache = None
        self.cache_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.capture_thread = None
        self.print_lock = threading.Lock()
        self.action_buffer = deque()

        cprint("[*] 控制器初始化完成", "green")
        self.start_capture_thread()

    def load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def safe_cprint(self, *args, **kwargs):
        with self.print_lock:
            cprint(*args, **kwargs)

    # ----------------------------------------------------------------
    # --- 1. 后台采集线程 (获取相机+机器人状态) ---
    # ----------------------------------------------------------------
    def start_capture_thread(self):
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_and_display_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def stop_capture_thread(self):
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        self.camera_manager.close()

    def _capture_and_display_loop(self):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        while not self.stop_event.is_set():
            try:
                # 1. 获取图像数据
                rgb_dict, _ = self.camera_manager.get_data()

                # 2. 获取机器人状态 (Qpos)
                # get_qpos 返回: [J1...J7, Gripper]
                qpos = self.robot.get_qpos().tolist()

                # 3. 数据打包
                files_to_send, data_to_send = self._pack_data(rgb_dict, qpos)

                with self.cache_lock:
                    self.latest_observation_cache = {
                        "files": files_to_send,
                        "data": data_to_send,
                        "raw_images": rgb_dict,
                    }

                # 4. 显示
                self._display_images(rgb_dict)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception as e:
                self.safe_cprint(f"[Error] Capture loop: {e}", "red")
                time.sleep(0.1)

        cv2.destroyAllWindows()

    def _pack_data(self, rgb_dict, qpos):
        files_to_send = []

        # 遍历本地相机数据，并重命名为服务器需要的名字
        for local_name, img in rgb_dict.items():
            server_name = self.camera_mapping.get(local_name, local_name)

            # 压缩为 JPEG
            img_bytes = self._image_to_jpeg_bytes(img)
            files_to_send.append(
                ("images", (f"{server_name}.jpg", img_bytes, "image/jpeg"))
            )

        data_to_send = {"qpos": json.dumps(qpos)}
        return files_to_send, data_to_send

    def _image_to_jpeg_bytes(self, image):
        # 简单的 JPEG 压缩
        ret, buf = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
        )
        return buf.tobytes()

    def _display_images(self, raw_images):
        display_list = []
        for name in self.display_order:
            if name in raw_images:
                # 这里的 raw_images 已经是 RGB (来自 CameraManager)
                # cv2.imshow 需要 BGR
                bgr = cv2.cvtColor(raw_images[name], cv2.COLOR_RGB2BGR)

                # 统一高度，方便拼接
                target_h = 360
                h, w = bgr.shape[:2]
                scale = target_h / h
                resized = cv2.resize(bgr, (int(w * scale), target_h))
                display_list.append(resized)

        if display_list:
            combined = np.hstack(display_list)
            cv2.imshow(self.WINDOW_NAME, combined)

    # ----------------------------------------------------------------
    # --- 2. 机器人控制与通信 ---
    # ----------------------------------------------------------------
    def get_observation_from_cache(self):
        with self.cache_lock:
            if self.latest_observation_cache is None:
                return None, None, None
            c = self.latest_observation_cache
            return c["files"], c["data"], c["raw_images"]

    def get_action(self):
        files, data, raw_imgs = self.get_observation_from_cache()
        if not files:
            return 0, [], None, None, "No Data"

        start_t = time.perf_counter()
        try:
            resp = self.session.post(
                self.server_url + "/predict",
                files=files,
                data=data,
                proxies={"http": None, "https": None},
            )
            cost_time = time.perf_counter() - start_t

            if resp.status_code == 200:
                res_json = resp.json()
                qpos = json.loads(data["qpos"])
                return cost_time, qpos, res_json.get("actions"), raw_imgs, None
            else:
                return (
                    cost_time,
                    [],
                    None,
                    raw_imgs,
                    f"Server Error: {resp.status_code}",
                )

        except Exception as e:
            return 0, [], None, raw_imgs, str(e)

    def run_control_loop(self):
        self.safe_cprint("\n[*] 启动 Kinova 远程推理控制...", "magenta")

        # 1. 激活机器人
        self.robot.enable()
        self.action_buffer.clear()

        cprint("[*] 按 'q' + 回车 停止控制", "yellow")

        step = 0
        try:
            while step < self.max_steps:
                start_loop_t = time.perf_counter()

                # 检查退出
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    if sys.stdin.readline().strip().lower() == "q":
                        break

                action = None

                # 策略: 如果缓存有动作，先执行缓存；否则请求新动作
                if self.action_buffer:
                    action = self.action_buffer.popleft()
                else:
                    _, _, actions, _, err = self.get_action()
                    if err:
                        self.safe_cprint(f"[Err] {err}", "red")
                    elif actions:
                        # 处理 Chunk Action (如果是二维列表)
                        if isinstance(actions[0], list):
                            self.action_buffer.extend(actions)
                            action = self.action_buffer.popleft()
                        else:
                            action = actions

                if action:
                    # [关键] 调用 Kinova 的执行接口
                    self.robot.exec_action(action)

                    step += 1
                    status = f"Step: {step} | Action: {action[:3]}..."
                    sys.stdout.write(f"\r{ERASE_LINE}🟢 {status}")
                    sys.stdout.flush()

                # 控频
                dt = time.perf_counter() - start_loop_t
                time.sleep(max(0, 1.0 / self.control_freq - dt))

        except KeyboardInterrupt:
            pass
        finally:
            self.safe_cprint("\n[*] 停止控制，复位机器人...", "yellow")
            self.robot.home()

    # ----------------------------------------------------------------
    # --- 3. 辅助功能 ---
    # ----------------------------------------------------------------
    def clear_cache(self):
        try:
            task_desc = input("    ➡️  输入新任务描述 (回车跳过): ").strip()
            payload = {"task_description": task_desc} if task_desc else {}

            resp = requests.post(self.server_url + "/clear_cache", json=payload)
            if resp.status_code == 200:
                cprint("✅ 缓存已清除", "green")
            else:
                cprint(f"❌ 失败: {resp.status_code}", "red")
        except Exception as e:
            cprint(f"❌ 异常: {e}", "red")

    def test_connection(self):
        try:
            resp = requests.get(self.server_url + "/health", timeout=2)
            if resp.status_code == 200:
                cprint("✅ 服务器连接正常", "green")
            else:
                cprint(f"❌ 服务器返回: {resp.status_code}", "red")
        except Exception as e:
            cprint(f"❌ 连接失败: {e}", "red")

    def shutdown(self):
        self.stop_capture_thread()
        if self.robot:
            self.robot.home()


def print_menu():
    print("\n================= Kinova Remote Client =================")
    print("  1️⃣  测试服务器连接")
    print("  2️⃣  清除模型缓存 (Reset / New Task)")
    print("  3️⃣  开始远程推理控制 🦾")
    print("  0️⃣  退出")
    print("========================================================")
    print("请输入选项: ", end="", flush=True)


def flush_input():
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="src/kinova_teleop/scripts/utils/config.yaml"
    )
    parser.add_argument("--ip", type=str, default="192.168.3.101")  # 服务器 IP
    parser.add_argument("--port", type=str, default="6160")
    args = parser.parse_args()

    client = None
    try:
        client = KinovaRemoteController(args.config, args.ip, args.port)

        # 等待第一帧
        while client.latest_observation_cache is None:
            time.sleep(0.1)

        while True:
            print_menu()
            flush_input()
            choice = input().strip()

            if choice == "0":
                break
            elif choice == "1":
                client.test_connection()
            elif choice == "2":
                client.clear_cache()
            elif choice == "3":
                client.run_control_loop()
            else:
                print("无效选项")

    except KeyboardInterrupt:
        print("\n退出...")
    except Exception as e:
        cprint(f"\n[Fatal Error] {e}", "red")
        import traceback

        traceback.print_exc()
    finally:
        if client:
            client.shutdown()


if __name__ == "__main__":
    rospy_needed = True  # RobotManager 依赖 ROS
    if rospy_needed:
        import rospy

        rospy.init_node("kinova_remote_client", anonymous=True)

    main()
