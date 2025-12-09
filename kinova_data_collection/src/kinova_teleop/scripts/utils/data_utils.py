#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import cv2
import numpy as np
import h5py
import rospy
import tempfile
import zlib
from mcap_ros1.writer import Writer as Ros1Writer
from std_msgs.msg import Float32MultiArray


class UnifiedSaver:
    def __init__(self, save_dir, save_format="hdf5"):
        self.save_dir = save_dir
        self.save_format = save_format.lower()

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_episode(self, data_dict, episode_idx):
        """统一保存入口"""
        if self.save_format == "mcap":
            return self._save_mcap(data_dict, episode_idx)
        elif self.save_format == "hdf5":
            return self._save_hdf5(data_dict, episode_idx)

    # ==========================
    # 方案 A: HDF5 (Data_Asst 标准兼容)
    # ==========================
    def _save_hdf5(self, data_dict, episode_idx):
        filename = f"{episode_idx}.hdf5"
        file_path = os.path.join(self.save_dir, filename)

        print(f"\n[Saver] Saving HDF5 to {filename} (GZIP Level 4 + Chunking)...")
        t0 = time.time()

        try:
            with h5py.File(file_path, "w") as f:
                f.attrs["sim"] = False

                # 按照 data_asst 的习惯，通常结构是:
                # /action
                # /observations/qpos
                # /observations/images/cam_name
                # /observations/depth/cam_name

                # 1. 写入 Action
                f.create_dataset("action", data=np.array(data_dict["action"]))
                if "action_gpos" in data_dict:
                    f.create_dataset(
                        "action_gpos", data=np.array(data_dict["action_gpos"])
                    )

                # 2. 写入 Observations 组
                obs_grp = f.create_group("observation")
                obs_grp.create_dataset(
                    "state", data=np.array(data_dict["observation"]["state"])
                )
                if "qpos_gpos" in data_dict["observation"]:
                    obs_grp.create_dataset(
                        "state_gpos",
                        data=np.array(data_dict["observation"]["state_gpos"]),
                    )
                if "velocity" in data_dict["observation"]:
                    f["observation"].create_dataset(
                        "velocity", data=np.array(data_dict["observation"]["velocity"])
                    )
                if "effort" in data_dict["observation"]:
                    f["observation"].create_dataset(
                        "effort", data=np.array(data_dict["observation"]["effort"])
                    )

                # 3. 写入 Images (带压缩和分块)
                if "images" in data_dict["observation"]:
                    img_grp = obs_grp.create_group("images")
                    for cam_name, frames in data_dict["observation"]["images"].items():
                        arr = np.array(frames)  # Shape: (T, H, W, 3)
                        if len(arr) > 0:
                            img_grp.create_dataset(
                                cam_name,
                                data=arr,
                                dtype=np.uint8,
                                compression="gzip",
                                compression_opts=4,
                                chunks=(1, *arr.shape[1:]),  # (1, H, W, 3)
                            )

                # 4. 写入 Depth (如果有)
                if "depth" in data_dict["observation"]:
                    depth_grp = obs_grp.create_group("depth")
                    for cam_name, frames in data_dict["observation"]["depth"].items():
                        arr = np.array(frames)
                        if len(arr) > 0:
                            depth_grp.create_dataset(
                                cam_name,
                                data=arr,
                                compression="gzip",
                                compression_opts=4,
                                chunks=(1, *arr.shape[1:]),
                            )

            dt = time.time() - t0
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ Saved HDF5: {size_mb:.2f} MB in {dt:.2f}s")
            return True

        except Exception as e:
            print(f"❌ Error Saving HDF5: {e}")
            import traceback

            traceback.print_exc()
            return False

    # ==========================================
    # 方案 B: MCAP (Video + Zlib Attachments)
    # ==========================================
    def _save_mcap(self, data_dict, episode_idx):
        filename = f"{episode_idx}.mcap"
        file_path = os.path.join(self.save_dir, filename)
        print(f"\n[Saver] Encoding MCAP (Video/Zlib) to {filename}...")
        t0 = time.time()

        try:
            with open(file_path, "wb") as f:
                writer = Ros1Writer(output=f)

                # --- 1. 写入数据流 (Action/Qpos) ---
                length = len(data_dict["action"])
                base_time = rospy.Time.now() if rospy.get_rostime() else rospy.Time(0)

                for i in range(length):
                    t_now = base_time + rospy.Duration(i * 0.033)
                    t_ns = t_now.to_nsec()

                    # observation
                    qpos_msg = Float32MultiArray(
                        data=data_dict["observation"]["state"][i]
                    )
                    writer.write_message("/joint/state", qpos_msg, t_ns, t_ns)
                    if "state_gpos" in data_dict["observation"]:
                        qg_msg = Float32MultiArray(
                            data=data_dict["observation"]["state_gpos"][i]
                        )
                        writer.write_message("/eef/state_gpos", qg_msg, t_ns, t_ns)
                    if "velocity" in data_dict["observation"]:
                        vel_msg = Float32MultiArray(
                            data=data_dict["observation"]["velocity"][i]
                        )
                        writer.write_message("/joint/velocity", vel_msg, t_ns, t_ns)
                    if "effort" in data_dict["observation"]:
                        eff_msg = Float32MultiArray(
                            data=data_dict["observation"]["effort"][i]
                        )
                        writer.write_message("/joint/effort", eff_msg, t_ns, t_ns)

                    # Action
                    act_msg = Float32MultiArray(data=data_dict["action"][i])
                    writer.write_message("/joint/action", act_msg, t_ns, t_ns)
                    if "action_gpos" in data_dict:
                        ag_msg = Float32MultiArray(data=data_dict["action_gpos"][i])
                        writer.write_message("/eef/action_gpos", ag_msg, t_ns, t_ns)

                # --- 2. 写入 RGB 图像 (MP4 Attachment) ---
                if "images" in data_dict["observation"]:
                    for cam_name, frames in data_dict["observation"]["images"].items():
                        if len(frames) == 0:
                            continue

                        print(f"  -> Encoding RGB video for {cam_name}...")
                        video_bytes = self._encode_video_to_memory(frames, codec="mp4v")

                        if video_bytes:
                            # 命名：/camera_name/rgb/video
                            att_name = f"/{cam_name}/rgb/video"
                            writer._Writer__writer.add_attachment(
                                name=att_name,
                                log_time=base_time.to_nsec(),
                                create_time=base_time.to_nsec(),
                                data=video_bytes,
                                media_type="video/mp4",
                            )

                # --- 3. 写入深度图 (Zlib Compressed Attachment) ---
                # [核心修改] 将整个 Depth 数组打包压缩存入
                if "depth" in data_dict["observation"]:
                    for cam_name, frames in data_dict["observation"]["depth"].items():
                        if len(frames) == 0:
                            continue

                        # (1) 转为 Numpy 数组
                        depth_array = np.array(frames, dtype=np.uint16)
                        raw_size = depth_array.nbytes / (1024 * 1024)

                        # (2) Zlib 强压缩 (无损)
                        # level=4 是速度和压缩率的平衡点，深度图通常含有大量0，压缩率极高
                        compressed_bytes = zlib.compress(depth_array.tobytes(), level=4)
                        comp_size = len(compressed_bytes) / (1024 * 1024)

                        print(
                            f"  -> Compressing depth for {cam_name}: {raw_size:.2f}MB -> {comp_size:.2f}MB"
                        )

                        # (3) 存为二进制附件
                        # 命名：/camera_name/depth/zlib
                        # 之后在 data_asst 读取时，先 numpy.frombuffer 再 reshape 即可
                        att_name = f"/{cam_name}/depth/zlib_data"

                        writer._Writer__writer.add_attachment(
                            name=att_name,
                            log_time=base_time.to_nsec(),
                            create_time=base_time.to_nsec(),
                            data=compressed_bytes,
                            media_type="application/octet-stream",  # 通用二进制流
                        )

                writer.finish()

            dt = time.time() - t0
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✅ Saved MCAP: {size_mb:.2f} MB in {dt:.2f}s")
            return True

        except Exception as e:
            print(f"❌ Error Saving MCAP: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _encode_video_to_memory(self, frames, codec="mp4v"):
        """使用 OpenCV 将帧列表编码为 MP4 字节流"""
        if not frames:
            return None

        # 确保是 numpy array
        frames = np.array(frames)
        t, h, w, c = frames.shape

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            # [优化] 优先尝试 avc1 (H.264)，如果系统不支持则回退到 mp4v
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(tmp.name, fourcc, 30.0, (w, h))
                if not out.isOpened():
                    raise Exception("Codec not supported")
            except:
                # mp4v 是最兼容的，avc1 (H.264) 更小但需要系统安装了 openh264
                print(f"  [Warn] Codec {codec} failed, falling back to mp4v")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(tmp.name, fourcc, 30.0, (w, h))

            for i in range(t):
                # RGB -> BGR for OpenCV
                # frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
                # 适配求之那边的数采方案，直接把 RGB 当作 BGR 喂进去
                frame_bgr = frames[i]
                out.write(frame_bgr)
            out.release()

            # 重新读取文件字节
            tmp.seek(0)
            video_data = tmp.read()
            return video_data
