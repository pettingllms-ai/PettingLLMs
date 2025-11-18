#include <verilated.h>
#include "Vtop.h"         // Verilator 生成的头文件(假设顶层模块名为 "top")
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <regex>
#include <verilated_vcd_c.h>

// 如果需要波形输出，则可以包含
// #include <verilated_vcd_c.h>

//-----------------------------------------------------
// 根据实际顶层信号名进行修改/扩展
//-----------------------------------------------------

// 定义一个端口信息结构体
struct PortInfo {
    std::string name;
    int width;
    bool isInput;
};

// 在程序开始时解析端口信息
std::vector<PortInfo> parsePortsFromHeader() {
    std::vector<PortInfo> ports;

    // 可以通过解析Vtop.h或使用Verilator特定选项生成的端口信息文件
    // 这里只是一个示例，您需要根据实际情况实现

    std::ifstream headerFile("Vtop.h");
    if (!headerFile.is_open()) {
        std::cerr << "无法打开Vtop.h分析端口" << std::endl;
        return ports;
    }

    std::string line;
    std::regex inputPattern("VL_IN\\w*\\((\\w+),\\s*(\\d+)\\)");
    std::regex outputPattern("VL_OUT\\w*\\((\\w+),\\s*(\\d+)\\)");

    while (std::getline(headerFile, line)) {
        std::smatch match;
        if (std::regex_search(line, match, inputPattern)) {
            ports.push_back({match[1], std::stoi(match[2]), true});
        } else if (std::regex_search(line, match, outputPattern)) {
            ports.push_back({match[1], std::stoi(match[2]), false});
        }
    }

    return ports;
}

// 动态设置信号值
void setSignal(Vtop* dut, const std::string& sigName, int value, const std::vector<PortInfo>& ports) {
    static std::map<std::string, std::pair<void*, int>> portMap;

    // 第一次调用时初始化映射
    if (portMap.empty()) {
        for (const auto& port : ports) {
            if (port.isInput) {
                // 这里需要使用指针偏移或其他方法获取dut成员变量的地址
                // 简化示例，实际实现可能需要更复杂的方法
                void* addr = nullptr;

                if (port.name == "clk") addr = &(dut->clk);
                else if (port.name == "L") addr = &(dut->L);
                else if (port.name == "q_in") addr = &(dut->q_in);
                else if (port.name == "r_in") addr = &(dut->r_in);
                // 其他端口...

                if (addr) portMap[port.name] = {addr, port.width};
            }
        }
    }

    // 查找并设置信号值
    auto it = portMap.find(sigName);
    if (it != portMap.end()) {
        int mask = (1 << it->second.second) - 1;
        *(static_cast<CData*>(it->second.first)) = value & mask;
    } else {
        std::cerr << "未知输入信号: " << sigName << std::endl;
    }
}

// 动态获取信号值
int getSignal(Vtop* dut, const std::string& sigName, const std::vector<PortInfo>& ports) {
    static std::map<std::string, std::pair<void*, int>> portMap;

    // 第一次调用时初始化映射
    if (portMap.empty()) {
        for (const auto& port : ports) {
            if (!port.isInput) {
                void* addr = nullptr;

                if (port.name == "Q") addr = &(dut->Q);
                // 其他输出端口...

                if (addr) portMap[port.name] = {addr, port.width};
            }
        }
    }

    // 查找并获取信号值
    auto it = portMap.find(sigName);
    if (it != portMap.end()) {
        int mask = (1 << it->second.second) - 1;
        return *(static_cast<CData*>(it->second.first)) & mask;
    }

    return -1; // 未定义
}

//-----------------------------------------------------
// 组合逻辑测试（cmb）示例：
// - 无需关心时钟的跳变，只要一次性赋值输入 -> eval -> 读输出
// - 通常在实际中是对纯组合模块进行测试，或者只在同一"时刻"判断输出
//-----------------------------------------------------
void runCmbTest(Vtop* dut, const nlohmann::json& scenario, const std::vector<PortInfo>& ports) {
    std::cout << "[CMB TEST] Scenario: " << scenario["scenario"] << std::endl;

    // 从 JSON 中取出输入向量/输出向量数组
    auto inputArray  = scenario["input variable"];
    auto outputArray = scenario["output variable"];

    // 对组合测试而言，通常假设 inputArray.size() == outputArray.size() == 1
    // 或者只测试一组输入输出关系
    // 但这里演示一下"多组输入逐个测试"也行，只是相当于多次独立的组合测试
    for (size_t i = 0; i < inputArray.size(); i++) {
        // 1. 设置 DUT 输入
        for (auto it = inputArray[i].begin(); it != inputArray[i].end(); ++it) {
            std::string sigName = it.key();
            int value           = std::stoi(it.value().get<std::string>());
            setSignal(dut, sigName, value, ports);
        }

        // 2. 评估组合逻辑
        dut->eval();

        // 3. 获取 DUT 输出并和期望值比对
        if (i < outputArray.size()) {
            for (auto it = outputArray[i].begin(); it != outputArray[i].end(); ++it) {
                std::string outName = it.key();
                int expectedVal     = std::stoi(it.value().get<std::string>());
                int actualVal       = getSignal(dut, outName, ports);
                if (actualVal == expectedVal) {
                    std::cout << "  " << outName << " matched: " << actualVal << std::endl;
                } else {
                    std::cout << "  [ERROR] " << outName
                              << " mismatch! expected=" << expectedVal
                              << ", got=" << actualVal << std::endl;
                }
            }
        }
    }
}

//-----------------------------------------------------
// 时序逻辑测试（seq）示例：
// - 需要关心时钟 clk 在不同周期内的值（0->1->0->1 ...）
// - 常见流程：
//   * 在时钟上升沿（或下降沿）进行 latch/寄存器操作
//   * 调用 dut->eval() 并在必要时进行延时或者再 eval()
// - 根据 inputArray 和 outputArray 数组下标一一对应测试
//-----------------------------------------------------
void runSeqTest(Vtop* dut, const nlohmann::json& scenario, const std::vector<PortInfo>& ports) {
    std::cout << "[SEQ TEST] Scenario: " << scenario["scenario"] << std::endl;

    auto inputArray  = scenario["input variable"];
    auto outputArray = scenario["output variable"];

    // 假设 inputArray.size() == outputArray.size()，并且
    // 里面包含时钟的 0/1 跳变。这里的"4 个步骤"就代表 4 个时刻。
    // 当然你也可以更精细地做"半周期"级的模拟，这里演示简单做法。

    // 简单的时序仿真思路：对每一组输入：
    //   1. 将 inputVariable 赋给 DUT
    //   2. 执行 eval() => 该时刻下(或该沿后)可观察到输出
    //   3. 从 DUT 里读出输出，与期望输出比较
    //   4. 如果要严格模拟 clk 上下沿，可以再插入一个"半周期"的 clk 翻转
    //      并再次 eval()。根据你真实的电路需求来写。
    //   5. 进入下一组输入
    for (size_t i = 0; i < inputArray.size(); i++) {
        // 1. 设置 DUT 输入(包括 clk)
        for (auto it = inputArray[i].begin(); it != inputArray[i].end(); ++it) {
            std::string sigName = it.key();
            int value           = std::stoi(it.value().get<std::string>());
            setSignal(dut, sigName, value, ports);
        }

        // 2. 评估
        dut->eval();

        // 3. 输出比对
        if (i < outputArray.size()) {
            for (auto it = outputArray[i].begin(); it != outputArray[i].end(); ++it) {
                std::string outName = it.key();
                int expectedVal     = std::stoi(it.value().get<std::string>());
                int actualVal       = getSignal(dut, outName, ports);

                if (actualVal == expectedVal) {
                    std::cout << "  " << outName << " matched: " << actualVal << std::endl;
                } else {
                    std::cout << "  [ERROR] " << outName
                              << " mismatch! expected=" << expectedVal
                              << ", got=" << actualVal << std::endl;
                }
            }
        }
    }
}

//-----------------------------------------------------
// 主函数示例
// - 解析 JSON 文件
// - 对其中每个场景，先后调用 runCmbTest 或 runSeqTest
//   (也可以只调用一个，看你需要的测试类型)
//-----------------------------------------------------
int main(int argc, char test_vector_file, char type, char** argv) {
    Verilated::commandArgs(argc, argv);

    // 1. 读取 JSON 文件
    std::ifstream ifs(test_vector_file);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open test_vector.json" << std::endl;
        return -1;
    }

    nlohmann::json testData;
    ifs >> testData;

    // 2. 创建 DUT 实例
    Vtop* dut = new Vtop;

    // (如果需要波形输出，可在此处创建 VCD dump)
    // VerilatedVcdC* tfp = nullptr;
    // Verilated::traceEverOn(true);
    // dut->trace(tfp, 99);
    // tfp->open("waveform.vcd");

    // 解析端口信息
    auto ports = parsePortsFromHeader();

    // 3. 依次处理每个场景
    for (auto& scenario : testData) {
        // 这里示例：既演示 cmb 又演示 seq
        // 你可以根据实际需要仅保留 runSeqTest 或 runCmbTest
        // 也可以在 JSON 里多加一个标记字段比如 "type": "seq" / "cmb" 做不同调用

        if (type == "CMB") {
            runCmbTest(dut, scenario, ports);
        } else if (type == "SEQ") {
            runSeqTest(dut, scenario, ports);
        }

        // 为了让组合测试结束后的状态不影响时序测试，
        // 可以在这里将 dut 重置一下(如果你的电路可复位)或者重新 new 一个 Vtop
        delete dut;
        dut = new Vtop;

    }

    // if (tfp) { tfp->close(); }
    delete dut;
    return 0;
}
