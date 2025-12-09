#!/usr/bin/env python3
import subprocess
import glob
import os


def list_cameras():
    # 获取所有 video 设备
    video_devs = sorted(glob.glob("/dev/video*"))

    print(f"{'Device':<15} | {'ID_PATH (Use this in YAML)':<50} | {'ID_SERIAL'}")
    print("-" * 100)

    for dev in video_devs:
        try:
            # 使用 udevadm 获取设备信息
            cmd = f"udevadm info --query=all --name={dev}"
            output = subprocess.check_output(cmd, shell=True).decode("utf-8")

            id_path = ""
            id_serial = ""

            for line in output.split("\n"):
                if "ID_PATH=" in line:
                    id_path = line.split("=")[1]
                if "ID_SERIAL=" in line:
                    id_serial = line.split("=")[1]

            if id_path:
                print(f"{dev:<15} | {id_path:<50} | {id_serial}")
        except Exception:
            pass


if __name__ == "__main__":
    list_cameras()
