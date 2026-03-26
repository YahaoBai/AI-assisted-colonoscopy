#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>

// 核心数据结构转化：将目标位置(int)拆分为低字节和高字节
void IntToBytes(int num, unsigned char* bytes, int size) {
    for (int i = 0; i < size; i++) {
        int offset = i * 8;
        bytes[i] = (num >> offset) & 0xFF;
    }
}

// 封装 POSIX 原生串口初始化逻辑
int init_serial(const char* port) {
    // 以读写、非阻塞、不作为控制终端的模式打开设备节点
    int fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        return -1;
    }

    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        return -1;
    }

    // 绝对对齐物理通信波特率：115200
    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);

    // 配置 8N1 物理电气参数 (8数据位, 无校验, 1停止位)
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    
    // 禁用硬件流控，启用接收器并忽略调制解调器控制线
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;

    // 配置为纯原始数据透传模式 (Raw Mode)，禁用所有终端特殊字符处理
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
    tty.c_oflag &= ~OPOST;

    // 立即刷新并应用设置
    tcsetattr(fd, TCSANOW, &tty);
    return fd;
}

// 严格按照老师的 0x03 定点模式封装帧结构并下发
void send_position(int fd, unsigned char id, int target_pos) {
    // 基础帧结构：帧头(55 AA) 长度(04) ID 指令(03:无反馈定位) 索引(37) 数据段...
    unsigned char frame[9] = {0x55, 0xAA, 0x04, id, 0x03, 0x37, 0x00, 0x00, 0x00};
    unsigned char TmpTgt[2] = {0};

    // 写入目标位置的高低字节
    IntToBytes(target_pos, TmpTgt, 2);
    frame[6] = TmpTgt[0];
    frame[7] = TmpTgt[1];

    // 计算校验和：累加第 2 到 7 字节
    unsigned short sum = 0;
    for (unsigned int i = 2; i < 8; i++) {
        sum += frame[i];
    }
    frame[8] = (unsigned char)(sum & 0xFF);

    // 调用底层系统调用下发数据包
    write(fd, frame, 9);
    
    // 注释说明：因为使用的是 0x03 (无反馈模式)，电缸不会回复。
    // 因此在连续下发多电机指令时，无需添加 sleep() 阻塞延时，不会发生总线碰撞。
}

int main() {
    const char* port_name = "/dev/ttyUSB0";
    int fd = init_serial(port_name);
    
    if (fd < 0) {
        std::cerr << "致命错误：无法打开或配置串口 " << port_name << std::endl;
        return -1;
    }
    
    std::cout << "==== 物理通信链路初始化成功 ====" << std::endl;
    std::cout << "端口: " << port_name << " | 波特率: 115200" << std::endl;

    while (true) {
        int p1, p2, p3, p4;
        std::cout << "\n请输入 4 台电缸的目标位置 (十进制 0~2000，空格分隔，输入 -1 退出): ";
        std::cin >> p1;
        
        if (p1 == -1) {
            std::cout << "安全退出控制程序。" << std::endl;
            break;
        }
        
        std::cin >> p2 >> p3 >> p4;

        // 连续向多机总线下发驱动指令
        send_position(fd, 0x01, p1);
        send_position(fd, 0x02, p2);
        send_position(fd, 0x03, p3);
        send_position(fd, 0x04, p4);

        std::cout << "-> 运动控制指令群已下发总线。" << std::endl;
    }

    close(fd);
    return 0;
}