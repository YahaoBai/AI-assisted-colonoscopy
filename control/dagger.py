import mujoco
import mujoco.viewer
import numpy as np
import time
from pynput import keyboard
import os
import cv2
import math
from collections import deque
from PIL import Image

# ==========================================
# 1. 导入感知模块 (U-Net) 与 决策模块 (Policy)
# ==========================================
try:
    from perception.lumen_center_api import init as init_unet, get_lumen_center
    init_unet('./checkpoints/attention_best_model.pth')
    print(">>> U-Net 视觉感知模块初始化成功。")
except ImportError as e:
    print(f"⚠️ 致命错误: 无法导入 U-Net ({e})")
    exit()

try:
    from perception.api import predict
    print(">>> Policy (模仿学习策略) 初始化成功。")
except ImportError as e:
    print(f"⚠️ 致命错误: 无法导入 Policy ({e})")
    exit()

# ==========================================
# 2. 模糊 PID 专家控制器 (带积分分离)
# ==========================================
class FuzzyPIDController:
    def __init__(self, base_kp, base_ki, base_kd, limit=None):
        self.base_kp = base_kp
        self.base_ki = base_ki
        self.base_kd = base_kd
        self.kp = base_kp
        self.ki = base_ki
        self.kd = base_kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.limit = limit

    def _fuzzy_adapt_gains(self, error, delta_error):
        abs_error = abs(error)
        if abs_error > 0.15:
            self.ki = 0.0
            self.integral = 0.0 
        else:
            self.ki = self.base_ki
            
        if abs_error > 0.2:
            self.kp = self.base_kp * 0.7
            self.kd = self.base_kd * 1.5
        elif 0.05 < abs_error <= 0.2:
            self.kp = self.base_kp
            self.kd = self.base_kd
        else:
            self.kp = self.base_kp * 2.0
            self.kd = self.base_kd * 1.0

    def update(self, current_val, target_val, dt):
        error = target_val - current_val
        delta_error = (error - self.prev_error) / dt if dt > 0 else 0
        self._fuzzy_adapt_gains(error, delta_error)
        self.integral += error * dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * delta_error)
        self.prev_error = error
        if self.limit is not None:
            output = np.clip(output, -self.limit, self.limit)
        return output

# ==========================================
# 3. 全局配置与按键监听
# ==========================================
IMG_WIDTH, IMG_HEIGHT = 256, 256
HALF_WIDTH, HALF_HEIGHT = IMG_WIDTH * 0.5, IMG_HEIGHT * 0.5
INV_HEIGHT = 1.0 / IMG_HEIGHT

MANUAL_MOVE_SPEED = 0.6
MANUAL_ROTATE_SPEED = 1.8
ZOOM_SPEED = 10.0

TARGET_SAMPLES = 1000  # 新增: 核心参数，目标采集总帧数

key_states = {
    'forward': False, 'pitch_up': False, 'pitch_down': False,
    'yaw_left': False, 'yaw_right': False, 'zoom_in': False, 'zoom_out': False,
    'autopilot_toggle_pressed': False, 'autopilot_on': False,
    'manual_takeover': False  # 手动强制接管标志
}

def on_press(key):
    if hasattr(key, 'char'):
        try:
            if key.char == '1': key_states['forward'] = True
            elif key.char == '+': key_states['zoom_in'] = True
            elif key.char == '-': key_states['zoom_out'] = True
            elif key.char == 't': key_states['manual_takeover'] = True  # T 键触发手动接管
            elif key.char == '5':
                if not key_states['autopilot_toggle_pressed']:
                    key_states['autopilot_toggle_pressed'] = True
                    key_states['autopilot_on'] = not key_states['autopilot_on']
                    state = f"开启 (目标: {TARGET_SAMPLES} 帧)" if key_states['autopilot_on'] else "关闭 (手动)"
                    print(f"\n>>> 自动导航采集模式已 {state}")
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
            elif key.char == 't': key_states['manual_takeover'] = False
            elif key.char == '5': key_states['autopilot_toggle_pressed'] = False
        except AttributeError: pass
    else:
        if key == keyboard.Key.up: key_states['pitch_up'] = False
        elif key == keyboard.Key.down: key_states['pitch_down'] = False
        elif key == keyboard.Key.left: key_states['yaw_left'] = False
        elif key == keyboard.Key.right: key_states['yaw_right'] = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def process_mask_for_policy(raw_mask):
    if raw_mask is None:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    raw_mask *= 255
    return raw_mask

# ==========================================
# 4. 初始化仿真环境与状态变量
# ==========================================
dataset_path = "./dagger_dataset"
if not os.path.exists(dataset_path): os.makedirs(dataset_path)

try:
    xml_path = './assets/xml/colon_scene.xml'
    model = mujoco.MjModel.from_xml_path(xml_path) if os.path.exists(xml_path) else mujoco.MjModel.from_xml_string("<mujoco><worldbody><body name='camera_rig' pos='0 0 0' mocap='true'><geom type='box' size='0.05 0.05 0.05' rgba='1 0 0 1'/><camera name='endoscope_cam' mode='fixed' fovy='90'/></body></worldbody></mujoco>")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    mocap_id = model.body('camera_rig').mocapid[0]
    camera_id = model.camera('endoscope_cam').id
except Exception as e:
    print(f"初始化失败: {e}")
    exit()

# 状态缓冲与滤波器
frame_buffer = deque(maxlen=3)
history_x = deque(maxlen=5)
history_y = deque(maxlen=5)

pid_yaw = FuzzyPIDController(base_kp=0.05, base_ki=0.05, base_kd=0, limit=2.1)
pid_pitch = FuzzyPIDController(base_kp=0.05, base_ki=0.05, base_kd=0, limit=2.1)

# --- 级联状态机变量定义 ---
# 1. 减速逻辑状态
is_slowing_down = False
speed_trigger_counter = 0     
speed_recovery_counter = 0    
TRIGGER_FRAMES = 5            
RECOVERY_FRAMES = 30

# 2. 接管逻辑状态 (DAgger 红线防线)
is_pid_taking_over = False

# 接管条件 1：绝对误差突破 0.30
takeover_trigger_counter = 0
TAKEOVER_FRAMES = 3

# 接管条件 2：误差在减速区连续发散
prev_total_norm_error = 0.0
error_increase_counter = 0
DIVERGENCE_FRAMES = 10

# 接管条件 3：长时间未恢复平稳状态
unstable_frames_counter = 0
TIMEOUT_FRAMES = 180

frame_count = 0
collected_samples = 0

print("\n" + "="*60)
print("  结肠镜 DAgger 自动采集系统启动 ")
print("  快捷键指令：")
print("  [5] 开启/关闭自动导航采集")
print("  [T] 手动强制触发 PID 接管")
print("="*60 + "\n")

renderer.update_scene(data, camera="endoscope_cam")
_, init_mask = get_lumen_center(renderer.render())
std_mask = process_mask_for_policy(init_mask)
for _ in range(3): frame_buffer.append(std_mask)

# ==========================================
# 5. 核心物理与控制循环
# ==========================================
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.fixedcamid = camera_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            step_pitch, step_yaw = 0.0, 0.0
            local_displacement = np.zeros(3)

            renderer.update_scene(data, camera="endoscope_cam")
            pixels = renderer.render()
            center_coords, raw_mask = get_lumen_center(pixels)
            
            current_mask = process_mask_for_policy(raw_mask)
            frame_buffer.append(current_mask)

            norm_x, norm_y, total_norm_error = 0, 0, 0

            if center_coords is not None:
                px_x, px_y = center_coords
                history_x.append((px_x - HALF_WIDTH) * INV_HEIGHT)
                history_y.append((px_y - HALF_HEIGHT) * INV_HEIGHT)
                filtered_norm_x = float(np.median(history_x))
                filtered_norm_y = float(np.median(history_y))
                total_norm_error = math.hypot(filtered_norm_x, filtered_norm_y)

            if key_states['autopilot_on'] and center_coords is not None:
                
                # --- A. 推理双边动作 ---
                mask_stack = np.array(frame_buffer)
                action_policy = predict(mask_stack)
                pol_yaw, pol_pitch = float(action_policy[0]), float(action_policy[1])
                
                exp_yaw = pid_yaw.update(filtered_norm_x, 0.0, dt)
                exp_pitch = pid_pitch.update(filtered_norm_y, 0.0, dt)

                # --- B. 评估接管条件 (Policy -> PID) ---
                
                # 条件 1: 绝对误差红线 (> 0.30 连续 3 帧)
                if total_norm_error > 0.30:
                    takeover_trigger_counter += 1
                else:
                    takeover_trigger_counter = 0
                cond_extreme_err = (takeover_trigger_counter >= TAKEOVER_FRAMES)

                # 条件 2: 误差在减速区连续发散 (大于 0.18 且严格单调递增连续 10 帧)
                if total_norm_error > prev_total_norm_error and total_norm_error > 0.18:
                    error_increase_counter += 1
                else:
                    error_increase_counter = 0
                cond_divergence = (error_increase_counter >= DIVERGENCE_FRAMES)

                # 条件 3: 长期无法恢复平稳 (在减速状态或高误差区滞留超过 180 帧)
                if total_norm_error > 0.18 or is_slowing_down:
                    unstable_frames_counter += 1
                else:
                    unstable_frames_counter = 0
                cond_timeout = (unstable_frames_counter >= TIMEOUT_FRAMES)

                # 条件 4: 人为手动接管干预
                cond_manual = key_states['manual_takeover']

                # 满足任意一项，立刻剥夺 Policy 控制权
                is_danger_now = cond_extreme_err or cond_divergence or cond_timeout or cond_manual

                if is_danger_now and not is_pid_taking_over:
                    is_pid_taking_over = True
                    if cond_manual:
                        reason = "手动干预"
                    elif cond_extreme_err:
                        reason = "误差破界"
                    elif cond_divergence:
                        reason = "连续发散"
                    else:
                        reason = "收敛超时"
                    print(f"[{frame_count}] 🚨 [{reason}] PID 强制接管！开启逐帧高频采集。")

                # --- C. 评估释放条件 (PID -> Policy) ---
                if is_pid_taking_over:
                    # 释放需满足：PID 将误差压回 0.05 以内 + 度过冷却期 + 用户未按住 T 键
                    if total_norm_error < 0.05 and speed_recovery_counter == 0 and not key_states['manual_takeover']:
                        is_pid_taking_over = False
                        takeover_trigger_counter = 0
                        error_increase_counter = 0
                        unstable_frames_counter = 0
                        print(f"[{frame_count}] ✅ 误差收敛且平稳，控制权交还 Policy。")

                # --- D. 动作指令路由 ---
                if is_pid_taking_over:
                    step_yaw, step_pitch = exp_yaw, exp_pitch
                else:
                    step_yaw, step_pitch = pol_yaw, pol_pitch

                # --- E. 纵向减速逻辑评估 (严禁修改) ---
                base_auto_speed = MANUAL_MOVE_SPEED * 0.85
                if total_norm_error > 0.18:
                    speed_trigger_counter += 1
                else:
                    speed_trigger_counter = 0

                if speed_trigger_counter >= TRIGGER_FRAMES:
                    current_speed = base_auto_speed * 0.4
                    speed_recovery_counter = RECOVERY_FRAMES  
                    if not is_slowing_down:
                        print(f"    ⚠️ [减速保护] 触发 0.18，推进速度降至 40% ({current_speed:.3f})")
                        is_slowing_down = True
                else:
                    if speed_recovery_counter > 0:
                        speed_recovery_counter -= 1
                        current_speed = base_auto_speed * 0.4  
                    else:
                        current_speed = base_auto_speed
                        if is_slowing_down:
                            print(f"    🟢 [航速恢复] 退出降速冷却期，推进速度恢复 ({current_speed:.3f})")
                            is_slowing_down = False
                            
                local_displacement[2] -= current_speed * dt

                # --- F. DAgger 差异化数据采集与落盘 ---
                should_save = False
                if is_pid_taking_over or is_danger_now:
                    should_save = True  # 100% 逐帧记录
                else:
                    if frame_count % 5 == 0:
                        should_save = True  # 30% 降采样
                        
                if should_save:
                    save_path = os.path.join(dataset_path, f"step_{collected_samples:06d}.npz")
                    np.savez_compressed(
                        save_path,
                        img=np.array(frame_buffer, dtype=np.uint8),
                        action_learner=np.array([pol_yaw, pol_pitch], dtype=np.float32),
                        action_expert=np.array([exp_yaw, exp_pitch], dtype=np.float32),
                        action_exec=np.array([step_yaw, step_pitch], dtype=np.float32),
                        is_recovery_segment=np.array([1 if is_pid_taking_over else 0], dtype=np.int8),
                        total_norm_error=np.array([total_norm_error], dtype=np.float32)
                    )
                    collected_samples += 1
                    
                    # 新增: 采集进度打印
                    if collected_samples % 100 == 0:
                        print(f"[采集进度] 已序列化 {collected_samples} / {TARGET_SAMPLES} 组特征与动作数据...")

                    # 新增: 达到目标采集数量后的自动截断逻辑
                    if collected_samples >= TARGET_SAMPLES:
                        key_states['autopilot_on'] = False 
                        is_pid_taking_over = False
                        is_slowing_down = False
                        step_yaw = 0.0
                        step_pitch = 0.0
                        local_displacement[2] = 0.0
                        
                        print("\n" + "="*60)
                        print(f"✅ 成功采集 {TARGET_SAMPLES} 组高质量 DAgger 样本。")
                        print(">>> 数据流记录已自动终止，系统切回手动闲置模式。")
                        print("="*60 + "\n")
                
                # 记录本帧误差，供下一帧评估发散率
                prev_total_norm_error = total_norm_error

            elif not key_states['autopilot_on']:
                if is_slowing_down: is_slowing_down = False
                speed_recovery_counter = 0
                speed_trigger_counter = 0
                is_pid_taking_over = False
                takeover_trigger_counter = 0
                error_increase_counter = 0
                unstable_frames_counter = 0

                if key_states['forward']: local_displacement[2] -= MANUAL_MOVE_SPEED * dt
                if key_states['pitch_up']: step_pitch += MANUAL_ROTATE_SPEED * dt
                if key_states['pitch_down']: step_pitch -= MANUAL_ROTATE_SPEED * dt
                if key_states['yaw_left']: step_yaw += MANUAL_ROTATE_SPEED * dt
                if key_states['yaw_right']: step_yaw -= MANUAL_ROTATE_SPEED * dt

            # 物理引擎执行与步进
            if step_pitch != 0 or step_yaw != 0:
                current_quat = data.mocap_quat[mocap_id].copy()
                pitch_quat, yaw_quat, d_quat = np.zeros(4), np.zeros(4), np.zeros(4)
                mujoco.mju_axisAngle2Quat(pitch_quat, np.array([1.0, 0.0, 0.0]), step_pitch)
                mujoco.mju_axisAngle2Quat(yaw_quat, np.array([0.0, 1.0, 0.0]), step_yaw)
                mujoco.mju_mulQuat(d_quat, yaw_quat, pitch_quat)
                mujoco.mju_mulQuat(data.mocap_quat[mocap_id], current_quat, d_quat)
                
            if local_displacement[2] != 0:
                world_disp = np.zeros(3)
                mujoco.mju_rotVecQuat(world_disp, local_displacement, data.mocap_quat[mocap_id])
                data.mocap_pos[mocap_id] += world_disp

            mujoco.mj_step(model, data)
            viewer.sync()
            frame_count += 1
            time.sleep(max(0, dt - (time.time() - step_start)))

finally:
    listener.stop()
    print(f"程序结束。本次共采集 DAgger 样本: {collected_samples} 帧")