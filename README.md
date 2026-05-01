# bookarm_control_py

图书馆机械臂的 Python 控制项目。

项目中 `BookArm` 负责高层运动学、关节限位检查和高层控制意图；`actuator` 负责把高层指令转换成 ESP32 可以接收的 JSON 指令，并通过串口发送到底层固件。

## 安装

建议使用已经配置好的 `bookarm-beiyu` conda 环境：

```powershell
conda activate bookarm-beiyu
pip install -e .
```

如果重新创建环境，需要安装 Pinocchio：

```powershell
conda create -n bookarm-beiyu python=3.11
conda activate bookarm-beiyu
conda install pinocchio -c conda-forge -y
pip install -e .
```

## 常用示例

### 正运动学示例

```powershell
conda run -n bookarm-beiyu python scripts/bookarm_fk_demo.py
```

该脚本直接在代码中定义关节角，然后调用 `BookArm.forward_kinematics_dict()` 计算末端位姿。

### 逆运动学示例

```powershell
conda run -n bookarm-beiyu python scripts/bookarm_ik_demo.py
```

该脚本直接在代码中定义目标末端位置和初始关节角，然后调用 `BookArm.inverse_kinematics()` 求解关节角。

### 零位读取测试

```powershell
conda run -n bookarm-beiyu python example/1.zero_and_read.py --port COM8
```

该脚本用于真实硬件测试：从 ESP32 读取当前关节角，检查机械臂是否接近零位，并用读取到的关节角计算当前末端位姿。

### 高层关节运动测试

```powershell
conda run -n bookarm-beiyu python example/2.check_arm_move.py --port COM8
```

该脚本用于测试高层关节角运动指令是否可以正常发送到真实机械臂。完整链路是：

```text
BookArm.move_joints_deg -> ArmActuator -> ESP32 JSON serial transport
```

脚本中默认使用一个确定合法的关节构型：

```python
DEFAULT_TARGET_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
```

如果需要测试其他构型，可以在脚本中修改 `DEFAULT_TARGET_DEG`，或者运行时传入：

```powershell
conda run -n bookarm-beiyu python example/2.check_arm_move.py --port COM8 --target-deg "0,10,20,0,0"
```

`BookArm.move_joints_deg()` 会在发送前根据 URDF 关节限位进行检查。如果目标关节角非法，脚本会直接报错，不会发送到底层 ESP32。

常用参数：

- `--port COM8`：指定 ESP32 串口。
- `--baud 921600`：指定串口波特率，默认 `921600`。
- `--wait-response`：发送运动指令后等待 ESP32 返回一条 JSON 响应。
- `--read-after`：发送运动指令后再次读取 ESP32 反馈角度，用于检查实际角度和目标角度的误差。
- `--settle-time 1.0`：使用 `--read-after` 时，发送指令后等待机械臂运动稳定的时间。
- `--dry-run`：只打印将要发送的 ESP32 JSON 指令，不打开串口，也不控制真实机械臂。

例如发送后读取反馈：

```powershell
conda run -n bookarm-beiyu python example/2.check_arm_move.py --port COM8 --read-after
```

## 注意事项

- 真实硬件测试前，请确认机械臂周围没有障碍物。
- 首次运动建议使用较小角度，确认电机方向和关节限位都正确。
- 如果串口被占用，可以先关闭其他串口调试工具，或者重新插拔 ESP32。
