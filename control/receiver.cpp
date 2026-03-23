//软件测试python发出的数据能否被C++准确解析

#include <iostream>
#include <fcntl.h>    // POSIX 文件控制定义
#include <unistd.h>   // UNIX 标准函数定义
#include <termios.h>  // POSIX 终端控制定义
#include <string>
#include <cstdio>

// 1. 初始化底层串口通信
int init_serial_port(const char* port_name) {
    // 以读写 (O_RDWR)、不作为控制终端 (O_NOCTTY)、非阻塞 (O_NDELAY) 模式打开设备节点
    int fd = open(port_name, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        std::cerr << "❌ 错误: 无法打开设备节点 " << port_name << std::endl;
        return -1;
    }

    // 配置串口底层参数 (虽然 socat 会忽略波特率，但为了兼容真实物理硬件，必须配置)
    struct termios options;
    tcgetattr(fd, &options);
    
    // 设置波特率为 115200
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);
    
    // 8N1 格式 (8位数据，无校验，1位停止位)
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    
    // 原始数据模式 (Raw Mode)，禁用终端字符的特殊处理
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    
    tcsetattr(fd, TCSANOW, &options);
    fcntl(fd, F_SETFL, FNDELAY); // 确保 read() 操作为非阻塞

    return fd;
}

int main() {
    // 绑定 socat 生成的接收端节点 (请确认是否仍为 /dev/pts/3)
    const char* PORT = "/dev/pts/4"; 
    int serial_fd = init_serial_port(PORT);

    if (serial_fd == -1) return -1;
    std::cout << "✅ 机器人 C++ 节点已启动，正在持续监听 " << PORT << "..." << std::endl;

    std::string buffer = "";
    char read_buf[256];
    bool estop_latched = false;

    // 2. 机器人实时控制主循环
    while (true) {
        // 从底层缓冲区读取数据
        int bytes_read = read(serial_fd, &read_buf, sizeof(read_buf));

        if (bytes_read > 0) {
            // 将读取到的字节存入字符串缓冲区
            buffer.append(read_buf, bytes_read);

            // 按照我们 Python 端设定的帧结束符 '\n' 进行粘包解析
            size_t pos;
            while ((pos = buffer.find('\n')) != std::string::npos) {
                // 提取完整的一帧指令
                std::string frame = buffer.substr(0, pos);
                buffer.erase(0, pos + 1);

                if (!frame.empty() && frame.back() == '\r') {
                    frame.pop_back();
                }
                if (frame.empty()) {
                    continue;
                }

                if (frame.rfind("ESTOP", 0) == 0) {
                    std::string reason = "UNKNOWN";
                    size_t comma = frame.find(',');
                    if (comma != std::string::npos && comma + 1 < frame.size()) {
                        reason = frame.substr(comma + 1);
                    }
                    estop_latched = true;
                    std::cout << "[硬件层急停] ESTOP 已锁存, reason=" << reason << std::endl;
                    continue;
                }

                if (frame == "RESET") {
                    estop_latched = false;
                    std::cout << "[硬件层复位] 急停锁存已清除, 可恢复执行 CMD" << std::endl;
                    continue;
                }

                if (frame.rfind("CMD,", 0) == 0) {
                    if (estop_latched) {
                        std::cout << "[硬件层丢弃] ESTOP 锁存中, 忽略指令: " << frame << std::endl;
                        continue;
                    }

                    float m1 = 0.0f, m2 = 0.0f, m3 = 0.0f, m4 = 0.0f;
                    if (sscanf(frame.c_str(), "CMD,%f,%f,%f,%f", &m1, &m2, &m3, &m4) == 4) {
                        std::cout << "[硬件层执行] MotorTarget(mm): "
                                  << "m1=" << m1 << ", "
                                  << "m2=" << m2 << ", "
                                  << "m3=" << m3 << ", "
                                  << "m4=" << m4 << std::endl;
                    } else {
                        std::cout << "[硬件层告警] CMD 解析失败: " << frame << std::endl;
                    }
                    continue;
                }

                std::cout << "[硬件层告警] 未知帧类型: " << frame << std::endl;
            }
        }
        // 模拟机器人底层 1000Hz 的控制周期，防止 while 循环占满单个 CPU 核心
        usleep(1000); 
    }

    close(serial_fd);
    return 0;
}
