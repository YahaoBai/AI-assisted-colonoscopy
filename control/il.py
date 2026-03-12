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

frame_buffer = deque(maxlen=3)

FILTER_WINDOW = 5
history_x = deque(maxlen=FILTER_WINDOW)
history_y = deque(maxlen=FILTER_WINDOW)

is_slowing_down = False
speed_trigger_counter = 0     
speed_recovery_counter = 0    
TRIGGER_FRAMES = 5            
RECOVERY_FRAMES = 30 

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
            
            time.sleep(max(0, dt - (time.time() - step_start)))

finally:
    listener.stop()
    print("程序结束。")