# Quick Start Guide

Get the Kinova Gen3 teleoperation system running in 5 minutes.

## Prerequisites Check

```bash
# Check ROS2 Humble
ros2 --version

# Check Python version (need >=3.8)
python3 --version

# Check gamepad connection
ls /dev/input/js*
```

## Step 1: Install Dependencies (2 min)

```bash
cd kinova_teleoperation

# Install Python packages
pip install numpy scipy h5py PyYAML inputs

# Install ROS2 packages (if not already installed)
sudo apt install ros-humble-rclpy \
                 ros-humble-sensor-msgs \
                 ros-humble-trajectory-msgs \
                 ros-humble-control-msgs \
                 python3-pykdl
```

## Step 2: Calibrate Table Height (1 min)

**CRITICAL**: Must do this before first run!

```bash
# Option A: With robot connected
python3 scripts/calibrate_table.py \
    --robot-name my_gen3 \
    --urdf ../ros2_kortex/kortex_description/arms/gen3/7dof/urdf/gen3_macro.xacro

# Option B: Manual measurement
python3 scripts/calibrate_table.py --manual
```

This creates `config/safety_params.yaml` with table height.

## Step 3: Configure Vision Pro IP (30 sec)

Edit `config/robot_config.yaml`:

```yaml
vision_pro:
  ip: "YOUR_VISION_PRO_IP"  # e.g., 192.168.1.100
```

Find IP on Vision Pro: Settings → Wi-Fi → (i) button

## Step 4: Start Robot Controller (30 sec)

```bash
# Terminal 1
ros2 launch kortex_bringup gen3.launch.py robot_ip:=YOUR_ROBOT_IP
```

Wait for "Robot ready" message.

## Step 5: Run Teleoperation System (30 sec)

```bash
# Terminal 2
cd kinova_teleoperation

python3 -m kinova_teleoperation.main_loop \
    --vision-pro-ip YOUR_VISION_PRO_IP \
    --robot-name my_gen3 \
    --config config/safety_params.yaml \
    --urdf ../ros2_kortex/kortex_description/arms/gen3/7dof/urdf/gen3_macro.xacro
```

## First Use

1. **Calibrate world frame**: Hold head still, press Enter when prompted
2. **Test clutch**: Press and hold A button on gamepad
3. **Move hand**: Robot should follow your hand motion
4. **Release clutch**: Let go of A button - robot holds position
5. **Control gripper**: Right trigger controls gripper (0=open, 1=closed)

## Controls Quick Reference

| Input | Action |
|-------|--------|
| A Button (hold) | Engage teleoperation |
| Y Button | Toggle fast/precision mode |
| Right Trigger | Gripper control |
| B Button | Emergency stop |
| Ctrl+C | Shutdown |

## Troubleshooting

### "Vision Pro not available"
```bash
# Test connection
ping YOUR_VISION_PRO_IP

# Test Vision Pro app is running
cd ../VisionProTeleop
python3 example.py --ip YOUR_VISION_PRO_IP
```

### "Gamepad not detected"
```bash
# Install gamepad tools
sudo apt install joystick

# Test gamepad
jstest /dev/input/js0
```

### "Robot not responding"
```bash
# Check robot controller is running
ros2 topic list | grep my_gen3

# Check joint states
ros2 topic echo /my_gen3/joint_states --once
```

## Test Without Hardware (Simulation Mode)

```bash
python3 -m kinova_teleoperation.main_loop \
    --vision-pro-ip 192.168.1.100 \
    --sim \
    --no-logging
```

This runs without requiring robot or Vision Pro (uses dummy data).

## Next Steps

- Collect demonstrations: Hold A button while performing task
- Demonstrations saved to `./demonstrations/*.hdf5`
- View statistics: System prints safety/IK stats on shutdown
- Adjust scaling: Edit `config/robot_config.yaml` scaling factors

## Need Help?

- Full documentation: [README.md](README.md)
- Run tests: `python3 tests/test_integration.py`
- Check logs: System prints detailed status to console
