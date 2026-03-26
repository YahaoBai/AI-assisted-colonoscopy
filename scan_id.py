import serial
import time

PORT = '/dev/ttyUSB0'
# 覆盖电缸支持的四种标准波特率
BAUDRATES = [921600, 115200, 57600, 19200]

def deep_scan():
    print(f"开始对 {PORT} 进行全波特率与 ID 深度扫描...\n")
    
    for baud in BAUDRATES:
        print(f"[*] 正在切换串口波特率至: {baud} bps")
        try:
            # 打开串口，禁用 DTR/RTS 避免触发某些 USB 芯片的硬件复位
            ser = serial.Serial(PORT, baud, timeout=0.05)
            ser.dtr = False
            ser.rts = False
            time.sleep(0.1) 
            
            # 扩大扫描范围至 ID 1~20
            for test_id in range(1, 21):
                # 构造查询状态指令: 55 AA 03 [ID] 04 00 22 [校验和]
                body = [0x03, test_id, 0x04, 0x00, 0x22]
                checksum = sum(body) & 0xFF
                cmd = bytes([0x55, 0xAA] + body + [checksum])
                
                ser.reset_input_buffer()
                ser.write(cmd)
                time.sleep(0.01) # 给 MCU 极短的响应时间
                
                if ser.in_waiting > 0:
                    response = ser.read(ser.in_waiting)
                    if b'\xaa\x55' in response:
                        print(f"\n==== 🎯 捕获成功 ====")
                        print(f"当前电缸的真实通信参数为 -> 波特率: {baud}, 硬件 ID: {test_id}")
                        print("====================\n")
                        ser.close()
                        return  # 找到后直接退出程序
            ser.close()
        except serial.SerialException as e:
            print(f"串口异常: {e}")
            
    print("\n深度扫描结束。未发现有效设备。")

if __name__ == "__main__":
    deep_scan()