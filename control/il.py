#具有减速逻辑的角度输出。
import mujoco
import mujoco.viewer
import numpy as np
import time
from pynput import keyboard
import os
import cv2
import math
from collections import deque

try:
    import serial
except ImportError:
    serial = None

# ==========================================
# 1. 导入感知模块 (Attention U-Net) 与 决策模块 (Policy)
# ==========================================
try:
    from perception.lumen_center_api import init as init_unet, get_lumen_center
    print(">>> 正在加载 Attention U-Net 感知模型...")
    init_unet('./checkpoints/attention_best_model.pth')
    print(">>> U-Net 初始化成功。")
except ImportError as e:
    print(f"⚠️ 致命错误: 无法导入 U-Net ({e})。请检查 lumen_center_api.py")
    exit()

try:
    from perception.api import predict
    print(">>> 成功导入 Imitation Learning 策略 API。")
except ImportError as e:
    print(f"⚠️ 致命错误: 无法导入 Policy ({e})。请检查 api.py")
    exit()

from control.sim2real_bridge import (
    BridgeCommand,
    DaggerSim2RealRuntimeConfig,
    Sim2RealBridge,
    load_dagger_sim2real_runtime_config,
)

# ==========================================
# 2. 全局配置与状态机
# ==========================================
IMG_WIDTH = 256
IMG_HEIGHT = 256

# [极速优化 3]: 提取常量，将循环内的除法转换为乘法
HALF_WIDTH = IMG_WIDTH * 0.5
HALF_HEIGHT = IMG_HEIGHT * 0.5
INV_HEIGHT = 1.0 / IMG_HEIGHT

MANUAL_MOVE_SPEED = 0.6
MANUAL_ROTATE_SPEED = 1.8
ZOOM_SPEED = 10.0

SIM2REAL_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "sim2real_config.yaml",
)

key_states = {
    'forward': False, 'pitch_up': False, 'pitch_down': False,
    'yaw_left': False, 'yaw_right': False, 'zoom_in': False, 'zoom_out': False,
    'autopilot_toggle_pressed': False, 'autopilot_on': False,
    'reset_estop_pressed': False,
}

def on_press(key):
    if hasattr(key, 'char'):
        try:
            if key.char == '1': key_states['forward'] = True
            elif key.char == '+': key_states['zoom_in'] = True
            elif key.char == '-': key_states['zoom_out'] = True
            elif key.char in ('r', 'R'):
                if not key_states['reset_estop_pressed']:
                    key_states['reset_estop_pressed'] = True
            elif key.char == '5':
                if not key_states['autopilot_toggle_pressed']:
                    key_states['autopilot_toggle_pressed'] = True
                    key_states['autopilot_on'] = not key_states['autopilot_on']
                    state = "开启 (混合控制: IL姿态 + 几何限速)" if key_states['autopilot_on'] else "关闭 (手动)"
                    print(f"\n>>> 自动导航模式已 {state}")
        except AttributeError: pass
    else:
        if key == keyboard.Key.up: key_states['pitch_up'] = True
        elif key == keyboard.Key.down: key_states['pitch_down'] = True
        elif key == keyboard.Key.left: key_states['yaw_left'] = True
        elif key == keyboard.Key.right: key_states['yaw_right'] = True

def on_release(key):
    if hasattr(key, 'char'):
        try:
            if key.char == '1': key_states['forward'] = False
            elif key.char == '+': key_states['zoom_in'] = False
            elif key.char == '-': key_states['zoom_out'] = False
            elif key.char in ('r', 'R'): key_states['reset_estop_pressed'] = False
            elif key.char == '5': key_states['autopilot_toggle_pressed'] = False
        except AttributeError: pass
    else:
        if key == keyboard.Key.up: key_states['pitch_up'] = False
        elif key == keyboard.Key.down: key_states['pitch_down'] = False
        elif key == keyboard.Key.left: key_states['yaw_left'] = False
        elif key == keyboard.Key.right: key_states['yaw_right'] = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# ==========================================
# 3. 初始化 MuJoCo 与时序滤波器
# ==========================================
try:
    if not os.path.exists('./assets/xml/colon_scene.xml'):
        model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body name="camera_rig" pos="0 0 0" mocap="true">
              <geom type="box" size="0.05 0.05 0.05" rgba="1 0 0 1"/>
              <camera name="endoscope_cam" mode="fixed" fovy="90"/>
            </body>
          </worldbody>
        </mujoco>
        """)
    else:
        model = mujoco.MjModel.from_xml_path('./assets/xml/colon_scene.xml')

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    mocap_id = model.body('camera_rig').mocapid[0]
    camera_id = model.camera('endoscope_cam').id
except Exception as e:
    print(f"初始化失败: {e}")
    listener.stop()
    exit()

# Sim2Real 配置加载（YAML）
try:
    runtime_cfg: DaggerSim2RealRuntimeConfig = load_dagger_sim2real_runtime_config(
        SIM2REAL_CONFIG_PATH
    )
    print(f">>> Sim2Real YAML 配置加载成功: {SIM2REAL_CONFIG_PATH}")
except Exception as e:
    print(f"❌ 致命错误: Sim2Real YAML 配置加载失败: {e}")
    print("❌ 为保证实机安全，程序拒绝启动。请修复 YAML 后重试。")
    listener.stop()
    raise SystemExit(1)

sim2real_cfg = runtime_cfg.bridge
estop_alarm_cfg = runtime_cfg.alarm
motor_output_cfg = runtime_cfg.output
sim2real_bridge = Sim2RealBridge(sim2real_cfg)

serial_link = None
if serial is None:
    print("❌ 致命错误: 未安装 pyserial，无法建立 Sim2Real 串口链路。")
    print("❌ 为保证实机安全，程序拒绝启动。")
    listener.stop()
    raise SystemExit(1)

try:
    serial_link = serial.Serial(
        sim2real_cfg.serial_port,
        baudrate=sim2real_cfg.serial_baudrate,
        timeout=sim2real_cfg.serial_timeout,
    )
    print(f">>> Sim2Real 串口链路已挂载: {sim2real_cfg.serial_port}")
except Exception as e:
    print(f"❌ 致命错误: Sim2Real 串口挂载失败: {e}")
    print("❌ 为保证实机安全，程序拒绝启动。")
    listener.stop()
    raise SystemExit(1)


def send_serial_frame(frame: str) -> bool:
    if serial_link is None:
        print(f"❌ [Sim2Real] 串口链路未就绪，拒绝发送: {frame.strip()}")
        return False

    payload = frame.encode("utf-8")

    try:
        bytes_written = serial_link.write(payload)
        serial_link.flush()
    except Exception as e:
        print(f"⚠️ 串口发送失败: {e} | frame={frame.strip()}")
        return False

    if bytes_written != len(payload):
        print(
            f"⚠️ 串口发送不完整: wrote={bytes_written} expected={len(payload)} | frame={frame.strip()}"
        )
        return False

    if motor_output_cfg.print_tx_frame:
        print(f"[Sim2Real TX] {frame.strip()}")

    return True


def send_critical_frame(frame: str, frame_name: str) -> bool:
    retry_count = int(max(1, sim2real_cfg.serial_critical_retry_count))
    retry_interval = float(max(0.0, sim2real_cfg.serial_critical_retry_interval_sec))

    for attempt in range(1, retry_count + 1):
        if send_serial_frame(frame):
            if attempt > 1:
                print(f"✅ [Sim2Real] 关键帧 {frame_name} 在第 {attempt} 次重发后成功。")
            return True

        print(f"⚠️ [Sim2Real] 关键帧 {frame_name} 发送失败（{attempt}/{retry_count}）。")
        if attempt < retry_count and retry_interval > 0.0:
            time.sleep(retry_interval)

    print(f"❌ [Sim2Real] 关键帧 {frame_name} 连续 {retry_count} 次发送失败: {frame.strip()}")
    return False


def emit_estop_alarm(frame_idx: int, reason: str, yaw_rad: float, pitch_rad: float) -> None:
    if not estop_alarm_cfg.enabled:
        return

    border = "!" * estop_alarm_cfg.banner_width
    bell = "\a" if estop_alarm_cfg.terminal_bell else ""
    alert_msg = (
        f"[ALARM][FRAME {frame_idx}] SIM2REAL ESTOP LATCHED | reason={reason} | "
        f"yaw={np.rad2deg(yaw_rad):.2f}deg | pitch={np.rad2deg(pitch_rad):.2f}deg | PRESS [R] TO RESET"
    )

    for _ in range(estop_alarm_cfg.repeat):
        print(f"{bell}{border}")
        print(alert_msg)
        print(border)


motor_frame_log = []
motor_target_log = []


def record_motor_target(frame_idx: int, motor_target_mm: np.ndarray) -> None:
    motor_frame_log.append(int(frame_idx))
    motor_target_log.append(np.asarray(motor_target_mm, dtype=np.float64).copy())

    if motor_output_cfg.print_every_n > 0:
        record_count = len(motor_target_log)
        if record_count % motor_output_cfg.print_every_n == 0:
            last = motor_target_log[-1]
            print(
                f"[MotorLog] 已记录 {record_count} 条电机目标位移 | "
                f"最新: m1={last[0]:.4f}, m2={last[1]:.4f}, m3={last[2]:.4f}, m4={last[3]:.4f} (mm)"
            )


def save_motor_outputs() -> None:
    if len(motor_target_log) == 0:
        print(">>> [MotorLog] 本次无可导出的电机目标位移记录。")
        return

    frames = np.asarray(motor_frame_log, dtype=np.int64)
    motors = np.asarray(motor_target_log, dtype=np.float64)

    if motor_output_cfg.save_csv:
        csv_path = motor_output_cfg.csv_path
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        csv_data = np.column_stack([frames, motors])
        np.savetxt(
            csv_path,
            csv_data,
            delimiter=",",
            header="frame,m1_target_mm,m2_target_mm,m3_target_mm,m4_target_mm",
            comments="",
            fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f"],
        )
        print(f">>> [MotorLog] 电机目标位移 CSV 已保存: {csv_path}")

    if motor_output_cfg.save_plot:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
        except ImportError:
            print("⚠️ [MotorLog] 未安装 matplotlib，跳过位移曲线图导出。")
            return

        # 优先尝试中文字体；若系统无可用中文字体则自动回退英文文案，避免乱码。
        cjk_font_candidates = [
            "Noto Sans CJK SC",
            "Noto Sans SC",
            "Microsoft YaHei",
            "SimHei",
            "PingFang SC",
            "WenQuanYi Zen Hei",
            "Source Han Sans SC",
            "Arial Unicode MS",
        ]
        available_font_names = {f.name for f in font_manager.fontManager.ttflist}
        chosen_cjk_font = next((name for name in cjk_font_candidates if name in available_font_names), None)
        if chosen_cjk_font is not None:
            matplotlib.rcParams["font.sans-serif"] = [
                chosen_cjk_font,
                *matplotlib.rcParams.get("font.sans-serif", []),
            ]
        matplotlib.rcParams["axes.unicode_minus"] = False

        if len(frames) > motor_output_cfg.max_plot_points:
            pick_idx = np.linspace(
                0,
                len(frames) - 1,
                num=motor_output_cfg.max_plot_points,
                dtype=np.int64,
            )
            frames_plot = frames[pick_idx]
            motors_plot = motors[pick_idx]
        else:
            frames_plot = frames
            motors_plot = motors

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frames_plot, motors_plot[:, 0], label="m1")
        ax.plot(frames_plot, motors_plot[:, 1], label="m2")
        ax.plot(frames_plot, motors_plot[:, 2], label="m3")
        ax.plot(frames_plot, motors_plot[:, 3], label="m4")
        if chosen_cjk_font is not None:
            ax.set_xlabel("帧号")
            ax.set_ylabel("电机绝对位移目标 (mm)")
            ax.set_title("电机绝对位移目标轨迹")
        else:
            ax.set_xlabel("Frame")
            ax.set_ylabel("Motor Absolute Target (mm)")
            ax.set_title("Motor Absolute Target Trajectory")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        plot_path = motor_output_cfg.plot_path
        plot_dir = os.path.dirname(plot_path)
        if plot_dir:
            os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(plot_path, dpi=motor_output_cfg.plot_dpi)
        plt.close(fig)
        print(f">>> [MotorLog] 电机目标位移曲线图已保存: {plot_path}")


frame_buffer = deque(maxlen=3)

FILTER_WINDOW = 5
history_x = deque(maxlen=FILTER_WINDOW)
history_y = deque(maxlen=FILTER_WINDOW)

is_slowing_down = False
speed_trigger_counter = 0     
speed_recovery_counter = 0    
TRIGGER_FRAMES = 5            
RECOVERY_FRAMES = 30 
frame_count = 0

# [极速优化 1] 废除耗时的 max() 扫描，直接就地内存乘法
def process_mask_for_policy(raw_mask):
    if raw_mask is None:
        return np.zeros((256, 256), dtype=np.uint8)
    # 明确知道 U-Net API 返回的是 uint8 格式的 0 和 1
    # 采用就地运算 (*=) 不分配新内存，速度达到物理极限
    raw_mask *= 255
    return raw_mask

print("\n" + "="*60)
print("     结肠镜仿真系统 ")
print("  [5] 自动模式开关 | [R] 清除角度锁存急停")
print("="*60 + "\n")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.fixedcamid = camera_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        
        # 预填充帧缓冲区
        renderer.update_scene(data, camera="endoscope_cam")
        _, init_mask = get_lumen_center(renderer.render())
        standard_mask = process_mask_for_policy(init_mask)
        for _ in range(3):
            frame_buffer.append(standard_mask)
            
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            if key_states['reset_estop_pressed']:
                key_states['reset_estop_pressed'] = False
                sim2real_bridge.reset()
                reset_frame, zero_frame = sim2real_bridge.get_reset_frames()
                reset_ok = send_critical_frame(reset_frame, "RESET")
                zero_ok = send_critical_frame(zero_frame, "CMD_ZERO")

                if reset_ok and zero_ok:
                    print(">>> [Sim2Real] 已清除急停锁存，累计角归零并下发零位。")
                else:
                    sim2real_bridge.estop_latched = True
                    key_states['autopilot_on'] = False
                    print("❌ [Sim2Real] RESET/回零下发失败，已保持锁存。请检查串口后按 [R] 重试。")

            if sim2real_bridge.estop_latched and key_states['autopilot_on']:
                key_states['autopilot_on'] = False
                print(">>> [Sim2Real] 角度急停已锁存，自动模式被禁止。请按 [R] 复位。")

            step_pitch = 0.0
            step_yaw = 0.0
            local_displacement = np.zeros(3)

            # 1. 渲染物理世界
            renderer.update_scene(data, camera="endoscope_cam")
            pixels = renderer.render()
            
            # --- 记录端到端推理管线的起始时间 ---
            pipeline_start = time.time()
            
            # 2. 感知层: 直接处理 NumPy 数组
            center_coords, raw_mask = get_lumen_center(pixels)
            
            # 3. 数据层: 极速张量映射
            current_mask = process_mask_for_policy(raw_mask)
            frame_buffer.append(current_mask)

            norm_x, norm_y, total_norm_error = 0, 0, 0

            # 4. 解算几何误差用于速度规划 (使用乘法替代除法)
            if center_coords is not None:
                px_x, px_y = center_coords
                raw_norm_x = (px_x - HALF_WIDTH) * INV_HEIGHT
                raw_norm_y = (px_y - HALF_HEIGHT) * INV_HEIGHT
                
                history_x.append(raw_norm_x)
                history_y.append(raw_norm_y)
                filtered_norm_x = float(np.median(history_x))
                filtered_norm_y = float(np.median(history_y))
                # 使用 math.hypot 代替 np.sqrt 标量运算更高效
                total_norm_error = math.hypot(filtered_norm_x, filtered_norm_y)

            # 5. 控制逻辑分支
            if key_states['autopilot_on']:
                if center_coords is not None:
                    # [决策网络]: 推理姿态增量
                    mask_stack = np.array(frame_buffer) 
                    action = predict(mask_stack) 
                    step_yaw = float(action[0])
                    step_pitch = float(action[1])
                    
                    # --- 记录结束时间并逐行打印延迟 ---
                    pipeline_end = time.time()
                    latency_ms = (pipeline_end - pipeline_start) * 1000
                    # 恢复普通逐行打印，方便后台挂机与日志记录
                    #print(f">>> [实时延迟] U-Net感知 -> Policy决策: {latency_ms:.2f} ms")

                    bridge_result = sim2real_bridge.step(step_yaw, step_pitch, dt)

                    if bridge_result.command == BridgeCommand.ESTOP:
                        estop_tx_ok = send_critical_frame(bridge_result.serial_frame, "ESTOP")
                        if not estop_tx_ok:
                            print("❌ [Sim2Real] ESTOP 下发失败，请立即人工确认底层处于安全状态。")
                        key_states['autopilot_on'] = False
                        is_slowing_down = False
                        speed_trigger_counter = 0
                        speed_recovery_counter = 0
                        step_yaw = 0.0
                        step_pitch = 0.0
                        local_displacement[2] = 0.0
                        emit_estop_alarm(
                            frame_idx=frame_count,
                            reason=bridge_result.reason or "ANGLE_LIMIT",
                            yaw_rad=bridge_result.yaw_accum_rad,
                            pitch_rad=bridge_result.pitch_accum_rad,
                        )
                        print(
                            f"[{frame_count}] 🛑 [Sim2Real] 角度越界触发急停锁存 | "
                            f"yaw={np.rad2deg(bridge_result.yaw_accum_rad):.2f}° "
                            f"pitch={np.rad2deg(bridge_result.pitch_accum_rad):.2f}° | 按 [R] 复位"
                        )
                    else:
                        if bridge_result.should_send and bridge_result.serial_frame:
                            if not send_serial_frame(bridge_result.serial_frame):
                                print("⚠️ [Sim2Real] CMD 下发失败，本帧目标未成功发送到底层。")
                            record_motor_target(frame_count, bridge_result.motor_target_mm)

                        # [状态机]: 动态降速逻辑与冷却保护
                        base_auto_speed = MANUAL_MOVE_SPEED * 0.85

                        if total_norm_error > 0.18:
                            speed_trigger_counter += 1
                        else:
                            speed_trigger_counter = 0

                        if speed_trigger_counter >= TRIGGER_FRAMES:
                            current_speed = base_auto_speed * 0.4
                            speed_recovery_counter = RECOVERY_FRAMES  
                            if not is_slowing_down:
                                # 移除开头的 \n，使日志输出紧凑整齐
                                print(f">>> [限速触发] 连续 {TRIGGER_FRAMES} 帧确认高曲率 | 推进速度降至: {current_speed:.3f}")
                                is_slowing_down = True
                        else:
                            if speed_recovery_counter > 0:
                                speed_recovery_counter -= 1
                                current_speed = base_auto_speed * 0.4  
                            else:
                                current_speed = base_auto_speed
                                if is_slowing_down:
                                    print(f">>> [限速解除] 误差回落且冷却期结束 | 推进速度恢复: {current_speed:.3f}")
                                    is_slowing_down = False
                                    
                        local_displacement[2] -= current_speed * dt
                else:
                    if is_slowing_down:
                        is_slowing_down = False
                    speed_recovery_counter = 0
                    speed_trigger_counter = 0
                    local_displacement[2] = 0.0
                    step_yaw = 0.0
                    step_pitch = 0.0
            else:
                # 手动模式逻辑
                if is_slowing_down:
                    is_slowing_down = False
                speed_recovery_counter = 0
                speed_trigger_counter = 0

                if key_states['forward']: local_displacement[2] -= MANUAL_MOVE_SPEED * dt
                if key_states['pitch_up']: step_pitch += MANUAL_ROTATE_SPEED * dt
                if key_states['pitch_down']: step_pitch -= MANUAL_ROTATE_SPEED * dt
                if key_states['yaw_left']: step_yaw += MANUAL_ROTATE_SPEED * dt
                if key_states['yaw_right']: step_yaw -= MANUAL_ROTATE_SPEED * dt

            # 6. 物理执行 (四元数运动学解算)
            if step_pitch != 0 or step_yaw != 0:
                current_quat = data.mocap_quat[mocap_id].copy()
                pitch_quat = np.zeros(4)
                mujoco.mju_axisAngle2Quat(pitch_quat, np.array([1.0, 0.0, 0.0]), step_pitch)
                yaw_quat = np.zeros(4)
                mujoco.mju_axisAngle2Quat(yaw_quat, np.array([0.0, 1.0, 0.0]), step_yaw)
                
                d_quat = np.zeros(4)
                mujoco.mju_mulQuat(d_quat, yaw_quat, pitch_quat)
                mujoco.mju_mulQuat(data.mocap_quat[mocap_id], current_quat, d_quat)
                
            if local_displacement[2] != 0:
                current_quat = data.mocap_quat[mocap_id].copy()
                world_disp = np.zeros(3)
                mujoco.mju_rotVecQuat(world_disp, local_displacement, current_quat)
                data.mocap_pos[mocap_id] += world_disp

            if key_states['zoom_in']: model.cam_fovy[camera_id] -= ZOOM_SPEED * dt
            if key_states['zoom_out']: model.cam_fovy[camera_id] += ZOOM_SPEED * dt

            mujoco.mj_step(model, data)
            viewer.sync()
            frame_count += 1
            
            time.sleep(max(0, dt - (time.time() - step_start)))

finally:
    listener.stop()
    if serial_link is not None:
        serial_link.close()
    save_motor_outputs()
    print("程序结束。")
