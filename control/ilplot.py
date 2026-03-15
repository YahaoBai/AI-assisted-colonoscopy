# 结肠镜仿真系统 - 适配 V3 CNN+ConvLSTM 版本 (集成双指标性能量化)
import mujoco
import mujoco.viewer
import numpy as np
import time
from pynput import keyboard
import os
import cv2
import math
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

# ==========================================
# 0. 轨迹量化评估器 (Trajectory Evaluator)
# ==========================================
class TrajectoryEvaluator:
    def __init__(self, output_dir="./eval_reports"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.is_recording = False
        self.weight_name = "v3_convlstm_policy" # 在对比测试时，可修改此名称区分权重
        self.reset_buffers()

    def reset_buffers(self):
        self.errors = []
        self.yaw_actions = []
        self.pitch_actions = []

    def start_recording(self):
        self.reset_buffers()
        self.is_recording = True
        print(f"\n[评估系统] 🔴 开始录制轨迹张量，当前测试权重: {self.weight_name}")

    def log_step(self, error, yaw, pitch):
        if self.is_recording:
            self.errors.append(error)
            self.yaw_actions.append(yaw)
            self.pitch_actions.append(pitch)

    def stop_and_report(self):
        if not self.is_recording or len(self.errors) < 10:
            self.is_recording = False
            return
            
        self.is_recording = False
        N = len(self.errors)
        print(f"\n[评估系统] ⏹️ 录制结束，共采集 {N} 帧闭环数据，正在计算核心量化指标...")

        # 1. 计算双核心指标
        errors_arr = np.array(self.errors)
        yaws_arr = np.array(self.yaw_actions)
        pitches_arr = np.array(self.pitch_actions)

        # 指标 A: 均方根误差 (RMSE)
        rmse = np.sqrt(np.mean(errors_arr**2))
        
        # 指标 B: 动作平滑度 (Action Smoothness, 一阶差分绝对值的均值)
        smoothness_yaw = np.mean(np.abs(np.diff(yaws_arr))) if N > 1 else 0
        smoothness_pitch = np.mean(np.abs(np.diff(pitches_arr))) if N > 1 else 0
        total_smoothness = smoothness_yaw + smoothness_pitch

        # 2. 终端输出报告
        print("="*50)
        print(f"      闭环性能量化报告 | 权重: {self.weight_name}")
        print("="*50)
        print(f" 轨迹总长度 (Frames)   : {N}")
        print(f" 均方根误差 (RMSE)     : {rmse:.4f}")
        print(f" 动作平滑度 (Smoothness): {total_smoothness:.4f}")
        print("="*50)

        # 3. 绘制并保存分析图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_prefix = f"{self.output_dir}/{self.weight_name}_{timestamp}"

        plt.figure(figsize=(12, 8))
        
        # 子图 1: 误差曲线
        plt.subplot(2, 1, 1)
        plt.plot(errors_arr, label='Tracking Error', color='red', alpha=0.8)
        plt.axhline(y=0.18, color='orange', linestyle='--', label='Slow-down Threshold (0.18)')
        plt.title(f'Tracking Error over Time (RMSE: {rmse:.4f})')
        plt.ylabel('Normalized Error')
        plt.legend()
        plt.grid(True)

        # 子图 2: 动作曲线
        plt.subplot(2, 1, 2)
        plt.plot(yaws_arr, label='Yaw Action', color='blue', alpha=0.6)
        plt.plot(pitches_arr, label='Pitch Action', color='green', alpha=0.6)
        plt.title(f'Action Output (Smoothness: {total_smoothness:.4f})')
        plt.xlabel('Frame')
        plt.ylabel('Action Value (rad)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = f"{report_prefix}_plot.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[评估系统] 📊 性能图表已保存至: {plot_path}")

# ==========================================
# 1. 导入感知模块 与 V3 决策模块 (CNN+ConvLSTM)
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
    from v3.api_convlstm3f import predict
    print(">>> 成功导入 V3 (CNN+ConvLSTM) 策略 API。")
except ImportError as e:
    print(f"⚠️ 致命错误: 无法导入 V3 Policy ({e})。请确保 v3/ 文件夹结构完整。")
    exit()

# ==========================================
# 2. 全局配置与状态机
# ==========================================
IMG_WIDTH = 256
IMG_HEIGHT = 256
HALF_WIDTH = IMG_WIDTH * 0.5
HALF_HEIGHT = IMG_HEIGHT * 0.5
INV_HEIGHT = 1.0 / IMG_HEIGHT

MANUAL_MOVE_SPEED = 0.6
MANUAL_ROTATE_SPEED = 1.8
ZOOM_SPEED = 10.0

# 实例化评估器
evaluator = TrajectoryEvaluator()

key_states = {
    'forward': False, 'pitch_up': False, 'pitch_down': False,
    'yaw_left': False, 'yaw_right': False, 'zoom_in': False, 'zoom_out': False,
    'autopilot_toggle_pressed': False, 'autopilot_on': False
}

def on_press(key):
    if hasattr(key, 'char'):
        try:
            if key.char == '1': key_states['forward'] = True
            elif key.char == '+': key_states['zoom_in'] = True
            elif key.char == '-': key_states['zoom_out'] = True
            elif key.char == '5':
                if not key_states['autopilot_toggle_pressed']:
                    key_states['autopilot_toggle_pressed'] = True
                    key_states['autopilot_on'] = not key_states['autopilot_on']
                    if key_states['autopilot_on']:
                        print(f"\n>>> 自动导航模式已 开启 (V3 混合控制: ConvLSTM + 几何限速)")
                        evaluator.start_recording() # 开启录制
                    else:
                        print(f"\n>>> 自动导航模式已 关闭 (手动)")
                        evaluator.stop_and_report() # 停止并计算报告
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
    xml_path = './assets/xml/colon_scene2.xml'
    if not os.path.exists(xml_path):
        xml_path = '../xml/colon_scene2.xml'
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    mocap_id = model.body('camera_rig').mocapid[0]
    camera_id = model.camera('endoscope_cam').id
except Exception as e:
    print(f"MuJoCo 初始化失败: {e}")
    listener.stop()
    exit()

frame_buffer = deque(maxlen=3)

FILTER_WINDOW = 5
history_x = deque(maxlen=FILTER_WINDOW)
history_y = deque(maxlen=FILTER_WINDOW)

is_slowing_down = False
speed_trigger_counter = 0     
speed_recovery_counter = 0    
TRIGGER_FRAMES = 5            
RECOVERY_FRAMES = 30 

def process_mask_for_policy(raw_mask):
    if raw_mask is None:
        return np.zeros((256, 256), dtype=np.uint8)
    raw_mask *= 255
    return raw_mask

print("\n" + "="*60)
print("     结肠镜仿真系统 (V3: ResNet18 + ConvLSTM | 性能量化版)")
print("="*60 + "\n")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.fixedcamid = camera_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        
        renderer.update_scene(data, camera="endoscope_cam")
        _, init_mask = get_lumen_center(renderer.render())
        standard_mask = process_mask_for_policy(init_mask)
        for _ in range(3):
            frame_buffer.append(standard_mask)
            
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            step_pitch = 0.0
            step_yaw = 0.0
            local_displacement = np.zeros(3)

            renderer.update_scene(data, camera="endoscope_cam")
            pixels = renderer.render()
            
            center_coords, raw_mask = get_lumen_center(pixels)
            
            current_mask = process_mask_for_policy(raw_mask)
            frame_buffer.append(current_mask)

            norm_x, norm_y, total_norm_error = 0, 0, 0

            if center_coords is not None:
                px_x, px_y = center_coords
                raw_norm_x = (px_x - HALF_WIDTH) * INV_HEIGHT
                raw_norm_y = (px_y - HALF_HEIGHT) * INV_HEIGHT
                
                history_x.append(raw_norm_x)
                history_y.append(raw_norm_y)
                filtered_norm_x = float(np.median(history_x))
                filtered_norm_y = float(np.median(history_y))
                total_norm_error = math.hypot(filtered_norm_x, filtered_norm_y)

            if key_states['autopilot_on']:
                if center_coords is not None:
                    mask_stack = np.array(frame_buffer) 
                    action = predict(mask_stack) 
                    step_yaw = float(action[0])
                    step_pitch = float(action[1])
                    
                    # 记录核心张量指标 (误差，横向偏移增量，纵向偏转增量)
                    evaluator.log_step(total_norm_error, step_yaw, step_pitch)

                    base_auto_speed = MANUAL_MOVE_SPEED * 0.85
                    
                    if total_norm_error > 0.18:
                        speed_trigger_counter += 1
                    else:
                        speed_trigger_counter = 0

                    if speed_trigger_counter >= TRIGGER_FRAMES:
                        current_speed = base_auto_speed * 0.4
                        speed_recovery_counter = RECOVERY_FRAMES  
                        if not is_slowing_down:
                            print(f">>> [限速触发] 连续 {TRIGGER_FRAMES} 帧高误差 | 航速切换至稳定模式")
                            is_slowing_down = True
                    else:
                        if speed_recovery_counter > 0:
                            speed_recovery_counter -= 1
                            current_speed = base_auto_speed * 0.4  
                        else:
                            current_speed = base_auto_speed
                            if is_slowing_down:
                                print(f">>> [限速解除] 误差回落 | 航速恢复正常")
                                is_slowing_down = False
                                
                    local_displacement[2] -= current_speed * dt
                else:
                    speed_recovery_counter = speed_trigger_counter = 0
                    local_displacement[2] = 0.0
                    step_yaw = step_pitch = 0.0
            else:
                if key_states['forward']: local_displacement[2] -= MANUAL_MOVE_SPEED * dt
                if key_states['pitch_up']: step_pitch += MANUAL_ROTATE_SPEED * dt
                if key_states['pitch_down']: step_pitch -= MANUAL_ROTATE_SPEED * dt
                if key_states['yaw_left']: step_yaw += MANUAL_ROTATE_SPEED * dt
                if key_states['yaw_right']: step_yaw -= MANUAL_ROTATE_SPEED * dt

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
            
            time.sleep(max(0, dt - (time.time() - step_start)))

finally:
    # 确保在异常中断或退出时也能保存已记录的数据
    evaluator.stop_and_report()
    listener.stop()
    print("仿真结束。")