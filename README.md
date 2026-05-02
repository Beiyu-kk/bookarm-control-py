# bookarm_control_py

图书馆机械臂的 Python 控制项目。

`BookArm` 负责高层机器人逻辑，包括 URDF 加载、正逆运动学、关节限位检查、机械臂运动、夹爪控制和反馈解析。底层 `actuator` 模块负责把高层命令转换成 ESP32 可以接收的 JSON 指令，并通过串口发送。

当前默认末端执行器 frame 是 `link5`。关节角在 `BookArm` 内部统一使用弧度。

## 快速开始

```powershell
conda activate bookarm-beiyu
pip install -e .
```

连接机械臂并移动到零位：

```python
from bookarm_control_py import BookArm

robot = BookArm()
robot.connect_serial_arm(port="COM8")

try:
    robot.move_zero_pose()
finally:
    robot.close()
```

## 文档

- [安装与环境](docs/0.installation.md)
- [示例脚本](docs/1.examples.md)
- [BookArm API](docs/2.api.md)
- [ESP32 JSON 指令](docs/3.esp32_protocol.md)

## 常用示例

```powershell
python example/0.check_arm_connection.py --port COM8
python example/0.check_arm_connection.py --port COM8 --skip-torque-test
python example/0.check_arm_connection.py --port COM8 --torque-test-delay 5.0
python example/1.zero_and_read.py --port COM8
python example/2.move_arm_to_q.py --port COM8
python example/3.check_fk_ik.py --port COM8
python example/4.check_ikine_best_effort.py --port COM8
python example/5.check_gripper.py --port COM8
python example/6.blind_grasp.py --port COM8
python scripts/goal_pose_grasp.py --port COM8
```

### best-effort IK 真机测试

`example/4.check_ikine_best_effort.py` 用于连接真实机械臂测试 `BookArm.ikine_best_effort`：

- 目标位姿在脚本 `main()` 中直接修改，`target_position` 单位为米，`target_rpy_deg` 单位为度。
- 姿态由 `bookarm_control_py.math_utils.rpy_to_matrix` 从欧拉角转换为旋转矩阵。
- 脚本连接机械臂后，会先移动到 `START_Q_DEG` 定义的起始构型。
- 到达起始构型并读取反馈后，使用该反馈关节角作为 IK 初始值 `q0`。
- `ikine_best_effort` 即使未达到容差，也会返回搜索过程中最接近目标的关节构型；脚本默认会移动到这个最近构型。
- 运动后会读取真实反馈，并打印关节误差、位置误差和姿态误差。

```powershell
python example/4.check_ikine_best_effort.py --port COM8
```

常用参数：

```powershell
python example/4.check_ikine_best_effort.py --port COM8 --speed 25 --acceleration 5 --arm-wait 5
python example/4.check_ikine_best_effort.py --port COM8 --max-iterations 500 --tolerance 1e-4 --step-size 0.4
```

### 固定目标位姿抓取

`scripts/goal_pose_grasp.py` 用于移动到固定目标位姿并执行抓取：

- 目标位姿在脚本 `main()` 中直接修改，`target_position` 直接给 xyz，`target_rpy_deg` 通过 `rpy_to_matrix` 转换为旋转矩阵。
- 逆解使用 `BookArm.ikine_best_effort`。
- 默认会移动到 best-effort 返回的最近构型，并继续执行抓取流程。
- 抓取后会先等待 `--grasp-hold-wait` 秒，再返回起始构型。
- 返回起始构型时速度单独由 `--return-speed` 控制，默认 `20.0`；其他机械臂运动速度默认仍为 `25.0`。

```powershell
python scripts/goal_pose_grasp.py --port COM8
```

常用参数：

```powershell
python scripts/goal_pose_grasp.py --port COM8 --speed 25 --return-speed 20
python scripts/goal_pose_grasp.py --port COM8 --grasp-hold-wait 3 --arm-wait 5 --gripper-wait 1
python scripts/goal_pose_grasp.py --port COM8 --no-close
```

## 注意事项

- 真实硬件运动前，确认机械臂周围没有障碍物。
- 第一次测试建议使用较小角度和较低速度。
- `BookArm` 会根据 URDF 检查关节角是否超限，超限会直接报错，不会自动裁剪。
- `ikine_best_effort` 的 `success=False` 不代表没有返回解，而是误差未达到 `tolerance`；脚本仍可能移动到返回的最近构型，请根据打印的位置误差和姿态误差判断目标是否合适。
- Windows 路径中包含中文时，Pinocchio 直接读取 URDF 可能失败。当前代码会保持 `DEFAULT_URDF_PATH` 指向 `assets` 下的 URDF，同时在内部复制到临时英文路径给 Pinocchio 加载。
- 如果串口被占用，请关闭串口调试助手、其他 Python 进程或重新插拔 ESP32。
- 如果机械臂和夹爪接在同一个 ESP32 串口上，优先使用 `robot.connect_serial(port="COM8")`，不要分别打开两次同一个 COM 口。
