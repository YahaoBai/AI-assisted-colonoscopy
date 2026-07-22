import mujoco
import mujoco.viewer
import numpy as np
import time
from pynput import keyboard
import os
from PIL import Image
import cv2
from collections import deque  

# ==========================================
# 1. 导入推理接口
# ==========================================
try:
    from perception.lumen_center_api import init, get_lumen_center
    print(">>> 正在加载 Attention U-Net 模型权重...")
    init('./checkpoints/attention_best_model.pth')
    print(">>> 模型初始化成功。")
except ImportError as e:
    print(f"⚠️ 警告: 导入失败 ({e})。请确认 attention_unet.py 和 lumen_center_api.py 都在当前目录下！")
    def get_lumen_center(image, return_time=False):
        fake_mask = np.zeros((image.size[1] if isinstance(image, Image.Image) else image.shape[0],
                              image.size[0] if isinstance(image, Image.Image) else image.shape[1]), dtype=np.uint8)
        return (256, 256), fake_mask

# ==========================================
# 2. 模糊 PID 控制器类 (已引入积分分离)
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
        self.last_output = 0.0

    def _fuzzy_adapt_gains(self, error, delta_error):
        abs_error = abs(error)
        
        # --------------------------------------------------
        # [核心新增] 积分分离：专治死区稳态误差，防过弯超调
        # --------------------------------------------------
        if abs_error > 0.15:
            self.ki = 0.0
            self.integral = 0.0  # 大误差时彻底清空积分，防止撞墙
        else:
            self.ki = self.base_ki   # 靠近中心时开启磁性吸附
            
        # --------------------------------------------------
        # 保持您完美手感的 Kp 与 Kd 分段参数绝对不动
        # --------------------------------------------------
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
            
        self.last_output = output
        return output

# ==========================================
# 3. 可视化 HUD 绘制函数 (已修正为所见即所得)
# ==========================================
def draw_debug_hud(overlay_pil, f_norm_x, f_norm_y, pid_yaw, pid_pitch):
    vis_img = np.array(overlay_pil)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    # 强制将 PID 真实计算的归一化误差映射回 256 尺度像素，确保连线完全对应底层逻辑
    c_x = int(f_norm_x * 256 + 128)
    c_y = int(f_norm_y * 256 + 128)
    t_x, t_y = 128, 128

    cv2.drawMarker(vis_img, (t_x, t_y), (0, 255, 0), cv2.MARKER_CROSS, 15, 1)
    cv2.line(vis_img, (t_x, t_y), (c_x, c_y), (0, 255, 255), 1)

    # 显示真实的归一化误差 nX/nY
    text_yaw = f"nX: {f_norm_x:.3f} | DY:{pid_yaw.last_output:.2f}"
    text_pit = f"nY: {f_norm_y:.3f} | DP:{pid_pitch.last_output:.2f}"
    
    cv2.rectangle(vis_img, (2, 2), (180, 45), (0, 0, 0), -1)
    cv2.putText(vis_img, text_yaw, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(vis_img, text_pit, (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return vis_img

# ==========================================
# 4. 全局配置与按键回调
# ==========================================
is_saving_debug = False
_cam_switch_cooldown = False
_light_switch_cooldown = False

IMG_WIDTH = 256
IMG_HEIGHT = 256

MANUAL_MOVE_SPEED = 0.4
MANUAL_ROTATE_SPEED = 1.8
ZOOM_SPEED = 10.0

key_states = {
    'forward': False, 'pitch_up': False, 'pitch_down': False,
    'yaw_left': False, 'yaw_right': False, 'zoom_in': False, 'zoom_out': False,
    'toggle_light_pressed': False, 'switch_cam_pressed': False,
    'autopilot_toggle_pressed': False, 'autopilot_on': False
}

def on_press(key):
    global is_saving_debug
    if hasattr(key, 'char'):
        try:
            if key.char == '1': key_states['forward'] = True
            elif key.char == '+': key_states['zoom_in'] = True
            elif key.char == '-': key_states['zoom_out'] = True
            elif key.char == '2':
                is_saving_debug = not is_saving_debug
                print(f"\n>>> {'开始' if is_saving_debug else '暂停'}保存 PID 调试图像 (至 ../masks 文件夹)...")
            elif key.char == '4':
                if not key_states['toggle_light_pressed']:
                    key_states['toggle_light_pressed'] = True
            elif key.char == '3':
                if not key_states['switch_cam_pressed']:
                    key_states['switch_cam_pressed'] = True
            elif key.char == '5':
                if not key_states['autopilot_toggle_pressed']:
                    key_states['autopilot_toggle_pressed'] = True
                    key_states['autopilot_on'] = not key_states['autopilot_on']
                    state = "开启 (Attention U-Net + PID)" if key_states['autopilot_on'] else "关闭 (手动)"
                    print(f"\n>>> 自动导航模式已 {state}")
        except AttributeError:
            pass
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
            elif key.char == '4': key_states['toggle_light_pressed'] = False
            elif key.char == '3': key_states['switch_cam_pressed'] = False
            elif key.char == '5': key_states['autopilot_toggle_pressed'] = False
        except AttributeError:
            pass
    else:
        if key == keyboard.Key.up: key_states['pitch_up'] = False
        elif key == keyboard.Key.down: key_states['pitch_down'] = False
        elif key == keyboard.Key.left: key_states['yaw_left'] = False
        elif key == keyboard.Key.right: key_states['yaw_right'] = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# ==========================================
# 5. 初始化 MuJoCo 与参数配置
# ==========================================
dataset_path = "./dataset"
if not os.path.exists(dataset_path): os.makedirs(dataset_path)
debug_path = "./masks"
if not os.path.exists(debug_path): os.makedirs(debug_path)

try:
    if not os.path.exists('./assets/xml/colon_scene2.xml'):
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
        model = mujoco.MjModel.from_xml_path('./assets/xml/colon_scene2.xml')

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    mocap_id = model.body('camera_rig').mocapid[0]
    camera_id = model.camera('endoscope_cam').id
except Exception as e:
    print(f"初始化失败: {e}")
    listener.stop()
    exit()

# [修改]: 保持 base_kp=0.05 不变，微量注入 base_ki=0.02 消除死区
pid_yaw = FuzzyPIDController(base_kp=0.05, base_ki=0.05, base_kd=0, limit=2.1)
pid_pitch = FuzzyPIDController(base_kp=0.05, base_ki=0.05, base_kd=0, limit=2.1)

frame_count = 0
infer_count = 0

# ==========================================
# 6. 主循环与离散状态机
# ==========================================
prev_yaw_angle = 0.0
prev_pitch_angle = 0.0

# 状态域中值滤波队列配置
FILTER_WINDOW = 5
history_x = deque(maxlen=FILTER_WINDOW)
history_y = deque(maxlen=FILTER_WINDOW)

# 双向蓄水池防抖振状态机
is_slowing_down = False
speed_trigger_counter = 0     
speed_recovery_counter = 0    
TRIGGER_FRAMES = 5            
RECOVERY_FRAMES = 30         

print("\n" + "="*60)
print("     结肠镜仿真系统 ")
print("="*60 + "\n")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.fixedcamid = camera_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            if key_states['switch_cam_pressed'] and not _cam_switch_cooldown:
                if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                else:
                    viewer.cam.fixedcamid = camera_id
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                _cam_switch_cooldown = True
            if not key_states['switch_cam_pressed']: _cam_switch_cooldown = False

            if key_states['toggle_light_pressed'] and not _light_switch_cooldown:
                try: model.vis.headlight.active = 1 - model.vis.headlight.active
                except: pass
                _light_switch_cooldown = True
            if not key_states['toggle_light_pressed']: _light_switch_cooldown = False

            step_pitch = 0.0
            step_yaw = 0.0
            local_displacement = np.zeros(3)

            renderer.update_scene(data, camera="endoscope_cam")
            pixels = renderer.render()
            pil_image = Image.fromarray(pixels)
            phys_pos = data.mocap_pos[mocap_id].copy()
            
            center_coords = None
            overlay_pil = None
            
            if key_states['autopilot_on'] or is_saving_debug:
                try:
                    result = get_lumen_center(pil_image)
                    center_coords, mask_img = result
                    if center_coords is not None:
                        overlay_np = np.array(pil_image).copy()
                        green_layer = np.zeros_like(overlay_np)
                        green_layer[:, :, 1] = 255
                        alpha = 0.4
                        mask_area = mask_img == 1
                        overlay_np[mask_area] = cv2.addWeighted(
                            overlay_np[mask_area], 1 - alpha,
                            green_layer[mask_area], alpha, 0
                        )
                        overlay_pil = Image.fromarray(overlay_np)
                except Exception as e:
                    pass

            norm_x, norm_y, total_norm_error = 0, 0, 0
            
            if center_coords is not None:
                px_x, px_y = center_coords
                raw_norm_x = (px_x - IMG_WIDTH / 2) / IMG_HEIGHT
                raw_norm_y = (px_y - IMG_HEIGHT / 2) / IMG_HEIGHT
                
                history_x.append(raw_norm_x)
                history_y.append(raw_norm_y)
                filtered_norm_x = float(np.median(history_x))
                filtered_norm_y = float(np.median(history_y))
                total_norm_error = np.sqrt(filtered_norm_x**2 + filtered_norm_y**2)

            if key_states['autopilot_on']:
                if center_coords is not None:
                    step_yaw = pid_yaw.update(filtered_norm_x, 0.0, dt)
                    step_pitch = pid_pitch.update(filtered_norm_y, 0.0, dt)
                    
                    yaw_velocity = abs(step_yaw / dt) if dt > 0 else 0.0
                    
                    prev_yaw_angle += step_yaw
                    prev_pitch_angle += step_pitch
                    
                    is_curve = False
                    if total_norm_error > 0.08 and (abs(prev_pitch_angle) > 0.5 or abs(prev_yaw_angle) > 0.5):
                        is_curve = True
                    elif abs(prev_pitch_angle) > 2.0:
                        is_curve = True
                    elif yaw_velocity > 1.5:
                        is_curve = True
                    
                    base_auto_speed = MANUAL_MOVE_SPEED * 0.85
                    
                    if total_norm_error > 0.18:
                        speed_trigger_counter += 1
                    else:
                        speed_trigger_counter = 0

                    if speed_trigger_counter >= TRIGGER_FRAMES:
                        current_speed = base_auto_speed * 0.4
                        speed_recovery_counter = RECOVERY_FRAMES  
                        if not is_slowing_down:
                            print(f">>> [降速保护] 连续 {TRIGGER_FRAMES} 帧确认大误差 | 航速降至: {current_speed:.3f}")
                            is_slowing_down = True
                    else:
                        if speed_recovery_counter > 0:
                            speed_recovery_counter -= 1
                            current_speed = base_auto_speed * 0.4  
                        else:
                            current_speed = base_auto_speed
                            if is_slowing_down:
                                print(f">>> [航速恢复] 误差回落且通过冷却期 | 航速恢复正常: {current_speed:.3f}")
                                is_slowing_down = False
                            
                    local_displacement[2] -= current_speed * dt

                    if overlay_pil and is_saving_debug:
                        should_save = False
                        if is_curve:
                            should_save = True  
                        else:
                            if infer_count % 3 == 0:
                                should_save = True  
                        
                        if should_save:
                            # 传入滤波后的归一化误差，实现所见即所得的视觉准星
                            hud = draw_debug_hud(overlay_pil, filtered_norm_x, filtered_norm_y, pid_yaw, pid_pitch)
                            cv2.imwrite(os.path.join(debug_path, f"debug_hud_{infer_count:05d}.jpg"), hud)
                        
                        infer_count += 1
                        if infer_count % 10 == 0:
                            print(f"正在采集中... 当前处理至 {infer_count:05d} 帧")
                else:
                    if is_slowing_down:
                        is_slowing_down = False
                    speed_recovery_counter = 0
                    speed_trigger_counter = 0
                        
                    local_displacement[2] = 0.0
                    step_yaw = 0.0
                    step_pitch = 0.0
                    
                    prev_yaw_angle += step_yaw
                    prev_pitch_angle += step_pitch
            else:
                if key_states['forward']: local_displacement[2] -= MANUAL_MOVE_SPEED * dt
                if key_states['pitch_up']: step_pitch += MANUAL_ROTATE_SPEED * dt
                if key_states['pitch_down']: step_pitch -= MANUAL_ROTATE_SPEED * dt
                if key_states['yaw_left']: step_yaw += MANUAL_ROTATE_SPEED * dt
                if key_states['yaw_right']: step_yaw -= MANUAL_ROTATE_SPEED * dt
                
                prev_yaw_angle += step_yaw
                prev_pitch_angle += step_pitch

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

            frame_count += 1
            viewer.sync()
            time.sleep(max(0, dt - (time.time() - step_start)))

finally:
    listener.stop()
    print("程序结束。")