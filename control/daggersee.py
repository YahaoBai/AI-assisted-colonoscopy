import os
import cv2
import numpy as np
import glob

# ==========================================
# 1. 配置参数
# ==========================================
DATASET_DIR = "./dagger_dataset1"  # 请确保与采集脚本的保存路径一致
FPS = 30                          # 回放帧率
DELAY_MS = int(1000 / FPS)

# ==========================================
# 2. 数据加载与排序
# ==========================================
if not os.path.exists(DATASET_DIR):
    print(f"❌ 错误: 找不到数据集目录 {DATASET_DIR}")
    exit()

# 获取所有 .npz 文件并按序列号严格排序
file_paths = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
file_paths.sort()

total_frames = len(file_paths)
if total_frames == 0:
    print(f"❌ 错误: 目录 {DATASET_DIR} 中没有找到 .npz 文件")
    exit()

print(f"✅ 成功加载 {total_frames} 帧数据。")
print("=" * 50)
print("  回放控制说明:")
print("  [空格键]  : 播放 / 暂停")
print("  [D] 键    : 暂停状态下，单帧前进")
print("  [A] 键    : 暂停状态下，单帧后退")
print("  [Q] 键    : 退出查看器")
print("=" * 50)

# ==========================================
# 3. 核心回放循环
# ==========================================
current_idx = 0
is_paused = False

cv2.namedWindow("DAgger Dataset Viewer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("DAgger Dataset Viewer", 512, 512)  # 放大显示以便于观察

while True:
    # 限制索引越界
    if current_idx >= total_frames:
        current_idx = total_frames - 1
        is_paused = True
    elif current_idx < 0:
        current_idx = 0

    # 1. 读取当前帧数据
    file_path = file_paths[current_idx]
    try:
        data = np.load(file_path)
        
        # 提取当前时刻的掩码 (时序张量的最后一帧)
        img_tensor = data['img']
        current_mask = img_tensor[-1]  # 取索引 2，即当前帧 m_t
        
        # 将单通道二值掩码转换为 BGR 三通道，以便绘制彩色 HUD
        vis_img = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
        
        # 提取动作与状态标量
        act_learner = data['action_learner']
        act_expert = data['action_expert']
        act_exec = data['action_exec']
        is_recovery = data['is_recovery_segment'][0]
        norm_error = data['total_norm_error'][0]
        
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        current_idx += 1
        continue

    # 2. 绘制状态指示器与 HUD 文本
    # 如果处于接管恢复段，为画面添加红色边框警示
    if is_recovery == 1:
        cv2.rectangle(vis_img, (0, 0), (255, 255), (0, 0, 255), 4)
        status_text = "STATUS: PID TAKEOVER (RECOVERY)"
        status_color = (0, 0, 255) # 红色
    else:
        status_text = "STATUS: POLICY CRUISING"
        status_color = (0, 255, 0) # 绿色

    # 构建文本信息
    texts = [
        f"Frame: {current_idx:05d} / {total_frames:05d}",
        status_text,
        f"Norm Error: {norm_error:.3f}",
        f"Learner Act: Y:{act_learner[0]:.2f} P:{act_learner[1]:.2f}",
        f"Expert Act : Y:{act_expert[0]:.2f} P:{act_expert[1]:.2f}",
        f"Exec Act   : Y:{act_exec[0]:.2f} P:{act_exec[1]:.2f}"
    ]

    # 添加半透明黑色背景板以提高文本可读性
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (5, 5), (230, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0, vis_img)

    # 逐行绘制文本
    y_offset = 18
    for i, text in enumerate(texts):
        color = status_color if i == 1 else (255, 255, 255)
        cv2.putText(vis_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        y_offset += 14

    # 3. 画面渲染
    cv2.imshow("DAgger Dataset Viewer", vis_img)

    # 4. 键盘状态机控制
    wait_time = 0 if is_paused else DELAY_MS
    key = cv2.waitKey(wait_time) & 0xFF

    if key == ord('q') or key == 27:  # Q 或 Esc 退出
        break
    elif key == ord(' '):  # 空格键切换播放/暂停
        is_paused = not is_paused
    elif key == ord('d'):  # D 键单帧前进
        current_idx += 1
        is_paused = True
    elif key == ord('a'):  # A 键单帧后退
        current_idx -= 1
        is_paused = True
    else:
        if not is_paused:
            current_idx += 1

cv2.destroyAllWindows()