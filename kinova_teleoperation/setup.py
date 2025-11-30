"""Setup file for kinova_teleoperation package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text()

setup(
    name="kinova_teleoperation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Teleoperation system for Kinova Gen3 using Vision Pro hand tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kinova-teleoperation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "h5py>=3.0.0",
        "PyYAML>=5.4.0",
        "inputs>=0.5",  # Gamepad library
    ],
    extras_require={
        "ros2": [
            # ROS2 dependencies (installed via apt)
            # "rclpy",
            # "sensor_msgs",
            # "trajectory_msgs",
            # "control_msgs",
        ],
        "kinematics": [
            "kdl_parser_py",
            "urdf_parser_py",
            # "PyKDL",  # Usually installed via apt: python3-pykdl
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "kinova-teleop=kinova_teleoperation.main_loop:main",
            "calibrate-table=scripts.calibrate_table:main",
        ],
    },
    include_package_data=True,
    package_data={
        "kinova_teleoperation": [
            "config/*.yaml",
            "launch/*.py",
        ],
    },
)
