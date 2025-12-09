#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import threading
import time
import subprocess
import glob
from abc import ABC, abstractmethod

import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a


class AbstractCamera(ABC):
    def __init__(self, cfg):
        self.name = cfg["name"]
        self.width = cfg.get("width", 640)
        self.height = cfg.get("height", 480)
        self.fps = cfg.get("fps", 30)
        self.use_depth = cfg.get("use_depth", False)

    @abstractmethod
    def read(self):
        """
        修改：返回 (ret, data_dict)
        data_dict = {'rgb': frame, 'depth': depth_frame(optional)}
        """
        pass

    @abstractmethod
    def close(self):
        pass


class OpenCVCamera(AbstractCamera):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.id_path = cfg["id"]  # yaml 里的 pci ID
        self.cap = None
        self.running = False
        self.lock = threading.Lock()

        # Buffer
        self.latest_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.grabbed = False

        self._connect()
        if self.cap and self.cap.isOpened():
            self.start()

    def _connect(self):
        # 1. 尝试寻找设备路径
        matched_dev = self._find_device_by_path(self.id_path)

        if matched_dev:
            # 提取数字索引 /dev/videoX -> X
            idx = int(matched_dev.replace("/dev/video", ""))
            self.cap = cv2.VideoCapture(idx)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                print(f"[Camera] {self.name} connected at {matched_dev}")
            else:
                print(f"[Camera] ❌ Error opening {matched_dev}")
        else:
            # 如果没找到，尝试直接用 id (万一是数字 0)
            if isinstance(self.id_path, int) or (
                isinstance(self.id_path, str) and self.id_path.isdigit()
            ):
                print(f"[Camera] Trying direct index {self.id_path} for {self.name}...")
                self.cap = cv2.VideoCapture(int(self.id_path))
            else:
                print(f"[Camera] ❌ Device ID {self.id_path} NOT FOUND")

    def _find_device_by_path(self, target_id_path):
        """通过 udevadm 扫描匹配 PCI ID"""
        if isinstance(target_id_path, int):
            return None

        video_devs = sorted(glob.glob("/dev/video*"))
        for dev in video_devs:
            try:
                cmd = f"udevadm info --query=all --name={dev} | grep ID_PATH="
                output = (
                    subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
                )
                if output:
                    current_id = output.split("=")[1]
                    if target_id_path in current_id:
                        return dev
            except:
                pass
        return None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.latest_rgb = frame
                        self.grabbed = True
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.grabbed, {
                "rgb": self.latest_rgb.copy(),
                "depth": None,  # USB 相机没有深度
            }

    def close(self):
        self.running = False
        if hasattr(self, "thread") and self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
        print(f"[Camera] {self.name} closed.")


class KinectCamera(AbstractCamera):
    def __init__(self, cfg):
        super().__init__(cfg)
        if pykinect is None:
            raise ImportError("pykinect_azure missing!")

        self.device_index = int(cfg.get("id", 0))
        self.depth_mode_str = cfg.get("depth_mode", "NFOV_UNBINNED")

        self.running = False
        self.lock = threading.Lock()

        # 目标尺寸
        self.target_w = self.width
        self.target_h = self.height

        # Buffer
        self.latest_rgb = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        self.latest_depth = np.zeros(
            (self.target_h, self.target_w), dtype=np.uint16
        )  # Depth是16位
        self.grabbed = False

        # 初始化 SDK
        pykinect.initialize_libraries()
        device_config = pykinect.default_configuration
        device_config.color_format = _k4a.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_720P  # 基底 720P

        # 配置深度模式
        if self.use_depth:
            if "WFOV" in self.depth_mode_str:
                device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
            else:
                device_config.depth_mode = _k4a.K4A_DEPTH_MODE_NFOV_UNBINNED
        else:
            device_config.depth_mode = _k4a.K4A_DEPTH_MODE_OFF

        self.device = pykinect.start_device(
            config=device_config, device_index=self.device_index
        )
        self.start()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            capture = self.device.update()  # 同时获取 RGB 和 Depth

            # 1. 获取 RGB
            ret_rgb, color_image = capture.get_color_image()
            if not ret_rgb:
                continue

            # 2. 获取 Depth (如果开启)
            depth_image = None
            if self.use_depth:
                # === 关键：获取“对齐到RGB”的深度图 ===
                # 这样深度图的分辨率和视角就和 RGB 完全一样了
                ret_depth, depth_image = capture.get_transformed_depth_image()
                if not ret_depth:
                    continue

            # 3. 裁剪逻辑 (Center Crop)
            img_h, img_w = color_image.shape[:2]
            start_x = (img_w - self.target_w) // 2
            start_y = (img_h - self.target_h) // 2

            # 裁剪 RGB (去 Alpha 通道)
            rgb_crop = color_image[
                start_y : start_y + self.target_h, start_x : start_x + self.target_w, :3
            ]

            # 裁剪 Depth
            depth_crop = None
            if self.use_depth and depth_image is not None:
                depth_crop = depth_image[
                    start_y : start_y + self.target_h, start_x : start_x + self.target_w
                ]

            # 存入 Buffer
            with self.lock:
                self.latest_rgb = np.ascontiguousarray(rgb_crop)
                if depth_crop is not None:
                    self.latest_depth = np.ascontiguousarray(depth_crop)
                self.grabbed = True

    def read(self):
        with self.lock:
            return self.grabbed, {
                "rgb": self.latest_rgb.copy(),
                # 如果没开启深度，返回 None
                "depth": self.latest_depth.copy() if self.use_depth else None,
            }

    def close(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()
        if hasattr(self, "device"):
            self.device.close()


class CameraManager:
    def __init__(self, camera_config_list):
        self.cameras = {}
        self.cam_list = []

        for cam_cfg in camera_config_list:
            c_type = cam_cfg.get("type")

            if c_type == "kinect":
                cam = KinectCamera(cam_cfg)
            elif c_type == "opencv":
                cam = OpenCVCamera(cam_cfg)
            else:
                print(f"[Camera] Unknown type: {c_type}")

            self.cameras[cam_cfg["name"]] = cam
            self.cam_list.append(cam)

            print(f"[*] Initialization {cam_cfg['name']} done...")

        # 显示线程
        self.display_running = False
        time.sleep(2.0)
        self.start_display()

    def start_display(self):
        self.display_running = True
        self.thread = threading.Thread(target=self._display_loop, daemon=True)
        self.thread.start()

    def _display_loop(self):
        while self.display_running:
            row_rgb_imgs = []
            row_depth_imgs = []

            # 标志位：这一帧里到底有没有任何一个相机产出了深度图？
            has_any_depth = False

            for cam in self.cam_list:
                ret, data = cam.read()
                rgb_frame = data["rgb"]
                depth_frame = data["depth"]

                # === 1. RGB 行 ===
                target_h = 480
                if rgb_frame.shape[0] != target_h:
                    scale = target_h / rgb_frame.shape[0]
                    rgb_frame = cv2.resize(
                        rgb_frame, (int(rgb_frame.shape[1] * scale), target_h)
                    )

                cv2.putText(
                    rgb_frame,
                    f"{cam.name} (RGB)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                row_rgb_imgs.append(rgb_frame)

                # === 2. Depth 行预处理 ===
                # 无论有没有深度，先准备数据，最后再决定要不要拼上去
                if depth_frame is not None:
                    has_any_depth = True  # <--- 发现有深度数据！

                    max_dist_mm = 2000.0
                    depth_vis = np.clip(depth_frame, 0, max_dist_mm)
                    depth_vis = (depth_vis / max_dist_mm * 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                    if depth_color.shape[:2] != rgb_frame.shape[:2]:
                        depth_color = cv2.resize(
                            depth_color, (rgb_frame.shape[1], rgb_frame.shape[0])
                        )

                    cv2.putText(
                        depth_color,
                        f"{cam.name} (Depth)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    row_depth_imgs.append(depth_color)
                else:
                    # 占位黑块
                    empty_block = np.zeros_like(rgb_frame)
                    # 只有当最终决定显示两行时，这个字才会被看到
                    cv2.putText(
                        empty_block,
                        "No Depth",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (100, 100, 100),
                        2,
                    )
                    row_depth_imgs.append(empty_block)

            # === 3. 智能拼接 ===
            if row_rgb_imgs:
                top_row = np.hstack(row_rgb_imgs)

                if has_any_depth:
                    # 只有当检测到至少有一个相机有深度时，才拼下面那一行
                    bottom_row = np.hstack(row_depth_imgs)
                    combined = np.vstack((top_row, bottom_row))
                else:
                    # 否则，只显示 RGB 这一行
                    combined = top_row

                # 缩放显示
                display_scale = 0.5
                small_view = cv2.resize(
                    combined,
                    (
                        int(combined.shape[1] * display_scale),
                        int(combined.shape[0] * display_scale),
                    ),
                )

                cv2.imshow("Robot Data Collection", small_view)
                if cv2.waitKey(30) == 27:
                    break
            else:
                time.sleep(0.1)

        cv2.destroyAllWindows()

    def get_data(self):
        """
        获取所有相机的数据，自动分类
        return: (rgb_dict, depth_dict)
        """
        rgb_dict = {}
        depth_dict = {}

        for name, cam in self.cameras.items():
            ret, data = cam.read()
            # 必须存 RGB
            rgb_dict[name] = data["rgb"]

            # 如果有深度，且不为 None，则存入 depth_dict
            if data["depth"] is not None:
                depth_dict[name] = data["depth"]

        return rgb_dict, depth_dict

    def close(self):
        self.display_running = False
        # thread join...
        for c in self.cameras.values():
            c.close()
        cv2.destroyAllWindows()
