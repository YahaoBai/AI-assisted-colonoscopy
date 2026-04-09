# AI-Assisted Colonoscopy Robot Navigation System

**Harbin Institute of Technology (Weihai)** | 


## 1. 工程概述 (Project Overview)

本项目为结肠镜机器人的视觉伺服与自动导航系统。工程实现了感知与控制模块的物理隔离与联合仿真，通过 MuJoCo 引擎构建高保真物理环境，结合 Attention U-Net 进行管腔中心提取，并利用积分分离模糊 PID 控制器输出高频姿态指令，同时训练Policy展开模仿学习。

## 2. 目录架构 (Directory Structure)

为保证代码的模块化与 `sys.path` 的正确解析，本工程严格遵循以下目录结构：

* `assets/`：存放 MuJoCo 仿真的静态资源（`.obj` 结肠网格模型、`.png` 贴图、`.xml` 场景配置）。
* `checkpoints/`：存放感知模块的网络权重（如 `.pth` 文件）。**注意：受 `.gitignore` 规则限制，此文件夹及权重文件不会被推送到远程仓库。**
* `control/`：存放底层控制算法，包括积分分离模糊 PID、C++ 串口通讯接收器 (`receiver.cpp`) 以及模仿学习 (`il.py`)。
* `perception/`：存放视觉感知模块，包含 Attention U-Net 网络架构、特征提取骨干网以及图像处理 API。

## 3. 核心控制逻辑 (Core Control Logic)

当前的自动驾驶混合控制状态机集成了动态降速与弯道检测机制。高曲率弯道的判定条件已精确固化为以下多维逻辑门：
* **条件 A：** 归一化中心误差 (`error`) > `0.08` 且 (俯仰角 `pitch` > `0.5` 或 偏航角 `yaw` > `0.5`)
* **条件 B：** 俯仰角速度 (`pitch angular velocity`) > `1.5`
* *(满足任意条件即触发减速与姿态修正策略)*

## 4. Git 协作规范 (Collaboration Workflow)

本项目的 `main` 分支为受保护的主干分支，仅存储经过验证的稳定代码。

1.  **物理隔离开发：** 任何新特性的开发（如更新 U-Net 结构或微调控制参数），必须在本地创建独立分支（如 `git switch -c feat-unet-update`）。
2.  **云端审查合并：** 本地测试通过后，将独立分支推送到 GitHub 远程仓库，并提交 Pull Request (PR)。经代码审查确认无路径冲突与接口异常后，方可合并入 `main`。

## 5. 手柄遥操作入口 (Teleop)

新增独立入口脚本：`control/teleop.py`（与 `control/il.py` 自动驾驶流程解耦）。

### 启动方式

```bash
python -m control.teleop
```

可选参数：
- `--dry-run`：仅验证手柄映射与安全状态机，不下发串口。
- `--debug-input`：低频打印手柄输入快照（轴值/按钮/角速度）。
- `--list-controllers`：扫描并列出当前识别到的手柄后退出。

### 按键约定（默认）

- 左摇杆：偏航/俯仰
- `Y`：前进占位状态（仅回调，不控制滑台）
- `A`:长按 1s：软件急停（ESTOP latch）
- `X`:（左侧按键）长按 1s：复位序列（fault_clear -> work_start -> follow_zero）
