# bookarm_control_py

图书馆 BookArm 机械臂的 Python 控制项目。项目提供三层能力：

- `BookArm`：高层机器人接口，负责 URDF 加载、正/逆运动学、关节限位检查、机械臂运动、夹爪控制和反馈解析。
- `actuator`：把高层机械臂/夹爪动作转换为 ESP32 固件使用的 JSON 指令。
- `camera` 与 `scripts/camera_arm`：封装 RealSense D435 RGB-D 相机，并支持点云选点后转换到机械臂基座坐标执行抓取。

当前默认末端执行器 frame 是 `link5`。`BookArm` 内部统一使用弧度表示机械臂关节角；示例脚本里为了方便阅读，常用角度数组定义目标构型，再转换为弧度。

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

如果机械臂和夹爪共用同一个 ESP32 串口，使用：

```python
robot.connect_serial(port="COM8")
```

## 文档

- [安装与环境](docs/0.installation.md)
- [示例脚本](docs/1.examples.md)
- [BookArm API](docs/2.api.md)
- [ESP32 JSON 指令](docs/3.esp32_protocol.md)
- [相机到机械臂外参](calibration/camera_to_base.json)

## 常用命令

以下命令都建议在项目根目录运行。

```powershell
python example/0.check_arm_connection.py --port COM8
python example/0.check_arm_connection.py --port COM8 --skip-torque-test
python example/1.zero_and_read.py --port COM8
python example/2.move_arm_to_q.py --port COM8
python example/3.check_fk_ik.py --port COM8
python example/4.check_ikine_best_effort.py --port COM8
python example/5.check_gripper.py --port COM8
python example/6.blind_grasp.py --port COM8

python scripts/arm/bookarm_fk_demo.py
python scripts/arm/bookarm_ik_demo.py
python scripts/arm/goal_pose_grasp.py --port COM8
python scripts/arm/manual_teach.py --port COM8

python scripts/camera/check_realsense_d435.py
python scripts/camera/view_realsense_point_cloud.py
python scripts/camera_arm/click_point_grasp.py
python scripts/camera_arm/click_point_grasp.py --port COM8 --execute
```

## 相机抓取

`scripts/camera_arm/click_point_grasp.py` 默认读取 `calibration/camera_to_base.json`，把 D435 相机坐标中的点转换为机械臂基座坐标：

```text
base_point_m = R_camera_to_base @ camera_point_m + xyz_m
```

建议先干跑，不连接机械臂：

```powershell
python scripts/camera_arm/click_point_grasp.py
```

确认打印出的相机点、基座点和 IK 结果合理后，再执行真实抓取：

```powershell
python scripts/camera_arm/click_point_grasp.py --port COM8 --execute
```

如果相机位置或朝向发生变化，必须更新 `calibration/camera_to_base.json`。文件中已经写了详细修改步骤和验证方法。

## 注意事项

- 真实硬件运动前，确认机械臂周围没有障碍物，并先用较低速度、小范围目标测试。
- `BookArm` 会根据 URDF 检查关节角是否超限，超限会直接报错，不会自动裁剪。
- `ikine_best_effort` 的 `success=False` 不代表没有解，而是误差未达到 `tolerance`；脚本仍会返回搜索过程中最接近目标的构型。
- Windows 路径中包含中文时，Pinocchio 直接读取 URDF 可能失败。当前代码会把 URDF 复制到临时英文路径后再加载。
- 如果串口被占用，请关闭串口调试助手、其他 Python 进程，或重新插拔 ESP32。
