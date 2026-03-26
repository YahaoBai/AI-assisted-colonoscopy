import serial
import time


PORT = "/dev/ttyUSB0"
BAUDRATE = 115200
ACTUATOR_IDS = [0x01, 0x02, 0x03, 0x04]
MIN_TARGET_POS = 0
MAX_TARGET_POS = 2000
INTER_CMD_DELAY_SEC = 0.002


def calc_checksum(frame_body):
    return sum(frame_body) & 0xFF


def build_position_frame(motor_id, target_pos):
    # 指令帧: 55 AA 04 ID 03 37 low high checksum
    low = target_pos & 0xFF
    high = (target_pos >> 8) & 0xFF
    frame_body = [0x04, motor_id, 0x03, 0x37, low, high]
    return bytes([0x55, 0xAA] + frame_body + [calc_checksum(frame_body)])


def build_single_control_frame(motor_id, cmd_value):
    # 单控指令帧: 55 AA 03 ID 04 00 CMD checksum
    frame_body = [0x03, motor_id, 0x04, 0x00, cmd_value]
    return bytes([0x55, 0xAA] + frame_body + [calc_checksum(frame_body)])


def init_serial():
    try:
        # 设置 write_timeout，避免串口异常时无限阻塞
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.05, write_timeout=0.2)
        ser.dtr = False
        ser.rts = False
        return ser
    except serial.SerialException as exc:
        print(f"致命错误：无法打开串口 {PORT}。错误信息: {exc}")
        return None


def safe_write(ser, frame, desc):
    try:
        ser.reset_input_buffer()
        written = ser.write(frame)
        ser.flush()
    except serial.SerialException as exc:
        print(f"[串口错误] {desc} 下发失败: {exc}")
        return False

    if written != len(frame):
        print(f"[串口错误] {desc} 短写: 期望 {len(frame)} 字节, 实际 {written} 字节")
        return False
    return True


def clamp_target_pos(raw_value):
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"位置值 {raw_value} 不是合法整数")

    if value < MIN_TARGET_POS:
        print(f"[安全裁剪] {value} < {MIN_TARGET_POS}，已裁剪为 {MIN_TARGET_POS}")
        return MIN_TARGET_POS
    if value > MAX_TARGET_POS:
        print(f"[安全裁剪] {value} > {MAX_TARGET_POS}，已裁剪为 {MAX_TARGET_POS}")
        return MAX_TARGET_POS
    return value


def send_position(ser, motor_id, target_pos):
    frame = build_position_frame(motor_id, target_pos)
    return safe_write(ser, frame, f"ID{motor_id} 位置指令")


def send_estop_all(ser):
    ok = True
    # 0x23: 急停
    for motor_id in ACTUATOR_IDS:
        frame = build_single_control_frame(motor_id, 0x23)
        if not safe_write(ser, frame, f"ID{motor_id} 急停"):
            ok = False
        time.sleep(INTER_CMD_DELAY_SEC)
    if ok:
        print("[安全] 已向全部电缸下发急停。")
    return ok


def send_work_start_all(ser):
    ok = True
    # 0x04: 工作启动
    for motor_id in ACTUATOR_IDS:
        frame = build_single_control_frame(motor_id, 0x04)
        if not safe_write(ser, frame, f"ID{motor_id} 工作启动"):
            ok = False
        time.sleep(INTER_CMD_DELAY_SEC)
    if ok:
        print("[控制] 已向全部电缸下发工作启动。")
    return ok


def main():
    ser = init_serial()
    if not ser:
        return

    print("==== 物理通信链路初始化成功 ====")
    print(f"端口: {PORT} | 波特率: {BAUDRATE}")
    print("模式: 0x03 (定位无反馈)")
    print("安全约束: 目标位置强裁剪到 [0, 2000]")
    print("命令: 输入4个整数下发位置; 输入 e 急停; 输入 r 工作启动; 输入 q 急停并退出")

    try:
        while True:
            user_input = input("\n请输入 4 台电缸目标位置: ").strip().lower()

            if user_input == "q":
                send_estop_all(ser)
                print("安全退出。")
                break

            if user_input == "e":
                send_estop_all(ser)
                continue

            if user_input == "r":
                send_work_start_all(ser)
                continue

            parts = user_input.split()
            if len(parts) != 4:
                print("错误：必须精确输入 4 个目标位置。")
                continue

            try:
                positions = [clamp_target_pos(v) for v in parts]
            except ValueError as exc:
                print(f"错误：{exc}")
                continue

            ok = True
            for motor_id, pos in zip(ACTUATOR_IDS, positions):
                if not send_position(ser, motor_id, pos):
                    ok = False
                    break
                time.sleep(INTER_CMD_DELAY_SEC)

            if not ok:
                print("[严重] 指令发送失败，立即触发全轴急停并退出。")
                send_estop_all(ser)
                break

            print(
                f"-> 已下发(裁剪后): ID1={positions[0]}, ID2={positions[1]}, "
                f"ID3={positions[2]}, ID4={positions[3]}"
            )

    except KeyboardInterrupt:
        print("\n程序被用户中断，正在急停...")
        send_estop_all(ser)
    finally:
        if ser and ser.is_open:
            ser.close()


if __name__ == "__main__":
    main()
