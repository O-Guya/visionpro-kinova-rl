# Kinova Gen3 Teleoperation System

Production-ready teleoperation system for Kinova Gen3 using Apple Vision Pro hand tracking and gamepad control for HIL-SERL demonstration data collection.

## Features

- **Real-time Control**: 20Hz control loop synchronized with Vision Pro streaming
- **Dual Input Modes**: Vision Pro hand tracking + Xbox/PlayStation gamepad
- **Comprehensive Safety**: Workspace limits, watchdog monitoring, and collision avoidance
- **Data Collection**: HDF5 recording with timestamp alignment for HIL-SERL
- **Advanced Filtering**: OneEuroFilter for smooth, low-latency tracking
- **Predictive Control**: 3-point trajectory extrapolation for reduced lag

## Hardware Requirements

- **Robot**: Kinova Gen3 7-DOF
- **Tracking**: Apple Vision Pro
- **Input**: Xbox or PlayStation gamepad (USB/Bluetooth)
- **Camera**: RealSense D435i (optional, for visual observations)
- **Computer**: Ubuntu 22.04 with ROS2 Humble

## System Architecture

### 6 Core Modules

1. **ReferenceFrameManager**: Transforms Vision Pro hand poses from head-relative to world-fixed coordinates with OneEuroFilter
2. **InputAggregator**: Unified gamepad input handling with edge detection and scaling modes
3. **SafetyMonitor**: Workspace constraints and system health watchdog
4. **MotionPlanner**: KDL-based IK solver with predictive trajectory generation
5. **DataLogger**: Multi-modal HDF5 recording with timestamp alignment
6. **RobotInterface**: ROS2 communication wrapper for Kinova Gen3

### Control Flow

```
Vision Pro (20Hz) → ReferenceFrameManager → Safety Check → IK Solver → Robot
                          ↓                        ↓            ↓
Gamepad Input → InputAggregator → Clutch Logic → Trajectory → DataLogger
```

## Installation

### 1. Install ROS2 Humble

```bash
# Follow official ROS2 Humble installation guide
# https://docs.ros.org/en/humble/Installation.html
```

### 2. Install Kinova ros2_kortex

```bash
cd ~/ros2_ws/src
# ros2_kortex should already be in your workspace
cd ~/ros2_ws
colcon build --packages-select kortex_bringup kortex_description
source install/setup.bash
```

### 3. Install Python Dependencies

```bash
cd kinova_teleoperation

# Install core dependencies
pip install -r requirements.txt

# Install ROS2 Python packages (via apt)
sudo apt install ros-humble-rclpy \
                 ros-humble-sensor-msgs \
                 ros-humble-trajectory-msgs \
                 ros-humble-control-msgs

# Install kinematics libraries
sudo apt install python3-pykdl
pip install kdl_parser_py urdf_parser_py

# Install package in development mode
pip install -e .
```

### 4. Install Vision Pro Streaming

```bash
# VisionProTeleop should already be in your repo
cd ../VisionProTeleop
pip install -e .
```

## Configuration

### 1. Calibrate Table Height

**CRITICAL**: You must calibrate the table height before running the system to ensure safety.

```bash
cd kinova_teleoperation

# With robot connected:
python3 scripts/calibrate_table.py \
    --robot-name my_gen3 \
    --urdf ../ros2_kortex/kortex_description/arms/gen3/7dof/urdf/gen3_macro.xacro

# Or manual calibration:
python3 scripts/calibrate_table.py --manual
```

This creates `config/safety_params.yaml` with measured workspace limits.

### 2. Update Vision Pro IP

Edit `config/robot_config.yaml`:

```yaml
vision_pro:
  ip: "192.168.1.XXX"  # Your Vision Pro IP
```

Find Vision Pro IP in Settings → Wi-Fi → (i) button.

### 3. Verify URDF Path

Edit `config/robot_config.yaml` to point to your Gen3 URDF:

```yaml
robot:
  urdf_path: "../ros2_kortex/kortex_description/arms/gen3/7dof/urdf/gen3_macro.xacro"
```

## Usage

### Quick Start

1. **Start the robot controller**:

```bash
# In terminal 1
ros2 launch kortex_bringup gen3.launch.py robot_ip:=192.168.1.10
```

2. **Connect Vision Pro**:
   - Launch the Vision Pro streaming app
   - Note the IP address displayed

3. **Connect gamepad**:
   - Plug in Xbox/PlayStation controller via USB or pair via Bluetooth
   - Test with `jstest /dev/input/js0`

4. **Run teleoperation system**:

```bash
# In terminal 2
cd kinova_teleoperation

python3 -m kinova_teleoperation.main_loop \
    --vision-pro-ip 192.168.1.XXX \
    --robot-name my_gen3 \
    --config config/safety_params.yaml \
    --urdf ../ros2_kortex/kortex_description/arms/gen3/7dof/urdf/gen3_macro.xacro
```

### Simulation Mode (Testing without Robot)

```bash
python3 -m kinova_teleoperation.main_loop \
    --vision-pro-ip 192.168.1.XXX \
    --sim \
    --no-logging
```

## Controls

### Gamepad Mapping (Xbox Controller)

- **A Button**: Clutch (hold to engage robot following)
- **Y Button**: Toggle scaling mode (Fast ↔ Precision)
- **Right Trigger**: Gripper control (0.0=open, 1.0=closed)
- **B Button**: Emergency stop
- **Ctrl+C**: Shutdown system

### Scaling Modes

- **Fast Mode**: XY=1.5×, Z=1.0× (default)
- **Precision Mode**: XY=0.5×, Z=0.5×

### Operation Workflow

1. **Calibration**: System prompts to calibrate world frame on startup
2. **Ready State**: Hold gamepad A button to engage clutch
3. **Teleoperation**: Move your hand while holding A to control robot
4. **Release**: Let go of A to disengage (robot holds position)
5. **Recording**: Demonstrations are automatically saved when clutch is engaged

## Data Collection

Demonstrations are saved to `./demonstrations/` in HDF5 format:

```
demonstrations/
  demo_20250130_143022.hdf5
  demo_20250130_143156.hdf5
  ...
```

### HDF5 Structure

```
observations/
  images: (N, 480, 640, 3) uint8
  qpos: (N, 7) float32          # Joint positions
  qvel: (N, 7) float32          # Joint velocities
  ee_pose: (N, 7) float32       # [x,y,z,qx,qy,qz,qw]

actions/
  cartesian_delta: (N, 6) float32  # [dx,dy,dz,drx,dry,drz]
  joint_positions: (N, 7) float32

metadata/
  task_name: str
  start_time: ISO timestamp
  num_frames: int
  ...
```

## Safety Features

### Watchdog Conditions (ANY triggers safety halt)

- Vision latency > 200ms
- Consecutive IK failures > 5
- Joint tracking error > 0.2 rad (jam detection)
- Tracking confidence drops below "High"

### Workspace Limits

- **Z-axis**: Minimum = `table_height_z + 3cm` (prevents table collision)
- **XY-plane**: Cylindrical boundary, radius = 60cm from robot base
- **Z-max**: 1.0m from base

Targets outside workspace are automatically clamped to safe region.

## Troubleshooting

### Vision Pro Connection Issues

```bash
# Test Vision Pro streaming
cd ../VisionProTeleop
python3 example.py --ip 192.168.1.XXX

# Check network connectivity
ping 192.168.1.XXX
```

### Gamepad Not Detected

```bash
# List available input devices
ls /dev/input/js*

# Test gamepad
sudo apt install joystick
jstest /dev/input/js0

# Install gamepad library
pip install inputs
```

### Robot Not Responding

```bash
# Check robot controller is running
ros2 topic list | grep my_gen3

# Check joint states
ros2 topic echo /my_gen3/joint_states

# Verify robot IP
ping 192.168.1.10  # Your robot IP
```

### IK Failures

- **Symptom**: "⚠ IK Failed" messages
- **Causes**:
  - Target pose unreachable (joint limits)
  - Target too far from current position
  - Singularity
- **Solution**: Move hand slower, stay within workspace

### Loop Overruns

- **Symptom**: "⚠ Loop overrun: XXXms"
- **Causes**: Computation taking > 50ms
- **Solutions**:
  - Disable trajectory extrapolation
  - Reduce image resolution
  - Close other applications

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run integration test
python3 tests/test_integration.py
```

### Module Testing

Each module has a `__main__` block for standalone testing:

```bash
# Test ReferenceFrameManager
python3 -m kinova_teleoperation.modules.reference_frame_manager

# Test InputAggregator
python3 -m kinova_teleoperation.modules.input_aggregator

# Test SafetyMonitor
python3 -m kinova_teleoperation.modules.safety_monitor
```

## Architecture Details

### OneEuroFilter Parameters

- `min_cutoff=1.0`: Minimum cutoff frequency (Hz)
- `beta=0.05`: Cutoff slope for velocity adaptation
- `d_cutoff=1.0`: Derivative filter cutoff

### Trajectory Extrapolation

Generates 3-point sliding window at t+50ms, t+100ms, t+150ms using hand velocity for linear prediction. Velocity clamped to 0.5 m/s maximum.

### Timestamp Alignment

- Δt < 50ms: Use directly
- 50ms ≤ Δt < 100ms: Linear interpolate
- Δt ≥ 100ms: Drop frame

## Citation

If you use this system in your research, please cite:

```bibtex
@software{kinova_teleoperation,
  title={Kinova Gen3 Teleoperation System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/kinova-teleoperation}
}
```

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please open an issue or pull request.

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/kinova-teleoperation/issues
- Email: your.email@example.com

## Acknowledgments

- Kinova Robotics for ros2_kortex package
- HIL-SERL for demonstration collection framework
- Apple Vision Pro for spatial tracking
