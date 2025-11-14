
#include "rfuzz-harness.h"
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <verilated.h>
#include "Vtop_module.h"
#include <sstream>

int fuzz_poke() {
    int unpass_total = 0;
    int unpass = 0;
    VerilatedContext* contextp;
    Vtop_module* top;

    VlWide<4> a_wide;
    VlWide<4> b_wide;
    VlWide<4> out_wide;
    // Scenario: SelectInputA_AllZeros0
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_0 {new VerilatedContext};
    contextp = contextp_0.get();
    top = new Vtop_module;
a_wide[0] = 0x00000000u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x00000000u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x00000000u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000000u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0xFFFFFFFFu;
 top->b[0]  = b_wide[0];
b_wide[1] = 0xFFFFFFFFu;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xFFFFFFFFu;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x0000000Fu;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x00000000u;
out_wide[1] = 0x00000000u;
out_wide[2] = 0x00000000u;
out_wide[3] = 0x00000000u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: SelectInputA_AllZeros0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "0");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario SelectInputA_AllZeros0" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario SelectInputA_AllZeros0" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: SelectInputB_AllOnes0
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_1 {new VerilatedContext};
    contextp = contextp_1.get();
    top = new Vtop_module;
a_wide[0] = 0x00000000u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x00000000u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x00000000u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000000u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0xFFFFFFFFu;
 top->b[0]  = b_wide[0];
b_wide[1] = 0xFFFFFFFFu;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xFFFFFFFFu;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x0000000Fu;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0xFFFFFFFFu;
out_wide[1] = 0xFFFFFFFFu;
out_wide[2] = 0xFFFFFFFFu;
out_wide[3] = 0x0000000Fu;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: SelectInputB_AllOnes0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "fffffffffffffffffffffffff");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario SelectInputB_AllOnes0" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario SelectInputB_AllOnes0" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: SelectInputA_Alternating0
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_2 {new VerilatedContext};
    contextp = contextp_2.get();
    top = new Vtop_module;
a_wide[0] = 0xAAAAAAAAu;
 top->a[0]  = a_wide[0];
a_wide[1] = 0xAAAAAAAAu;
 top->a[1]  = a_wide[1];
a_wide[2] = 0xAAAAAAAAu;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x0000000Au;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x00000000u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x00000000u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x00000000u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000000u;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0xAAAAAAAAu;
out_wide[1] = 0xAAAAAAAAu;
out_wide[2] = 0xAAAAAAAAu;
out_wide[3] = 0x0000000Au;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: SelectInputA_Alternating0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "1010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "aaaaaaaaaaaaaaaaaaaaaaaaa");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario SelectInputA_Alternating0" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario SelectInputA_Alternating0" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: SelectInputB_Alternating0
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_3 {new VerilatedContext};
    contextp = contextp_3.get();
    top = new Vtop_module;
a_wide[0] = 0x00000000u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x00000000u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x00000000u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000000u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0xAAAAAAAAu;
 top->b[0]  = b_wide[0];
b_wide[1] = 0xAAAAAAAAu;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xAAAAAAAAu;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x0000000Au;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0xAAAAAAAAu;
out_wide[1] = 0xAAAAAAAAu;
out_wide[2] = 0xAAAAAAAAu;
out_wide[3] = 0x0000000Au;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: SelectInputB_Alternating0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "aaaaaaaaaaaaaaaaaaaaaaaaa");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario SelectInputB_Alternating0" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario SelectInputB_Alternating0" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_00
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_4 {new VerilatedContext};
    contextp = contextp_4.get();
    top = new Vtop_module;
a_wide[0] = 0x1B4A93A5u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0xAD56897Cu;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x4664E8BDu;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000003u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x6BF2ECD0u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x6288AAE9u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x6C6FF6F1u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000000u;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x1B4A93A5u;
out_wide[1] = 0xAD56897Cu;
out_wide[2] = 0x4664E8BDu;
out_wide[3] = 0x00000003u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_00=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0011010001100110010011101000101111011010110101010110100010010111110000011011010010101001001110100101");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "0000011011000110111111110110111100010110001010001000101010101110100101101011111100101110110011010000");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "34664e8bdad56897c1b4a93a5");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_00" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_00" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_10
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_5 {new VerilatedContext};
    contextp = contextp_5.get();
    top = new Vtop_module;
a_wide[0] = 0xAD972406u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x55B66E17u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0xB04B8D6Du;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000003u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x57EDC03Au;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x5EB93858u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x04542D31u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000005u;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x57EDC03Au;
out_wide[1] = 0x5EB93858u;
out_wide[2] = 0x04542D31u;
out_wide[3] = 0x00000005u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0011101100000100101110001101011011010101010110110110011011100001011110101101100101110010010000000110");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "0101000001000101010000101101001100010101111010111001001110000101100001010111111011011100000000111010");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "504542d315eb9385857edc03a");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_10" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_10" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_20
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_6 {new VerilatedContext};
    contextp = contextp_6.get();
    top = new Vtop_module;
a_wide[0] = 0xDDC6F64Du;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x7B971278u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x23210BF8u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000006u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x2F7E199Eu;
 top->b[0]  = b_wide[0];
b_wide[1] = 0xD7C099FDu;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x5B67E497u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000007u;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0xDDC6F64Du;
out_wide[1] = 0x7B971278u;
out_wide[2] = 0x23210BF8u;
out_wide[3] = 0x00000006u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_20=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0110001000110010000100001011111110000111101110010111000100100111100011011101110001101111011001001101");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "0111010110110110011111100100100101111101011111000000100110011111110100101111011111100001100110011110");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "623210bf87b971278ddc6f64d");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_20" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_20" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_30
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_7 {new VerilatedContext};
    contextp = contextp_7.get();
    top = new Vtop_module;
a_wide[0] = 0x517F5C66u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0xCBC70A4Bu;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x8343F894u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000005u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0xD96A6663u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x8CCBFAD7u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xEC6F8052u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000009u;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0xD96A6663u;
out_wide[1] = 0x8CCBFAD7u;
out_wide[2] = 0xEC6F8052u;
out_wide[3] = 0x00000009u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_30=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0101100000110100001111111000100101001100101111000111000010100100101101010001011111110101110001100110");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1001111011000110111110000000010100101000110011001011111110101101011111011001011010100110011001100011");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "9ec6f80528ccbfad7d96a6663");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_30" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_30" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_40
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_8 {new VerilatedContext};
    contextp = contextp_8.get();
    top = new Vtop_module;
a_wide[0] = 0x2A81209Eu;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x87B8158Bu;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x80670EEFu;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x0000000Cu;
 top->a[3]  = a_wide[3];
b_wide[0] = 0xC059B048u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0xE6A74C3Du;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x7C73290Bu;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x0000000Du;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x2A81209Eu;
out_wide[1] = 0x87B8158Bu;
out_wide[2] = 0x80670EEFu;
out_wide[3] = 0x0000000Cu;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_40=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "1100100000000110011100001110111011111000011110111000000101011000101100101010100000010010000010011110");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1101011111000111001100101001000010111110011010100111010011000011110111000000010110011011000001001000");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "c80670eef87b8158b2a81209e");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_40" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_40" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_50
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_9 {new VerilatedContext};
    contextp = contextp_9.get();
    top = new Vtop_module;
a_wide[0] = 0x0C40D4F6u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0xC7861D94u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x1D8C7934u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000009u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x519605BCu;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x924BB906u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xC1A67ED2u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000009u;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x519605BCu;
out_wide[1] = 0x924BB906u;
out_wide[2] = 0xC1A67ED2u;
out_wide[3] = 0x00000009u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_50=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "1001000111011000110001111001001101001100011110000110000111011001010000001100010000001101010011110110");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1001110000011010011001111110110100101001001001001011101110010000011001010001100101100000010110111100");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "9c1a67ed2924bb906519605bc");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_50" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_50" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_60
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_10 {new VerilatedContext};
    contextp = contextp_10.get();
    top = new Vtop_module;
a_wide[0] = 0x5F3C8273u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x161A6156u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0xF9FABB5Au;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000003u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x6ABCBBA4u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x15A11684u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xB5297FE7u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x0000000Bu;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x5F3C8273u;
out_wide[1] = 0x161A6156u;
out_wide[2] = 0xF9FABB5Au;
out_wide[3] = 0x00000003u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_60=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0011111110011111101010111011010110100001011000011010011000010101011001011111001111001000001001110011");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1011101101010010100101111111111001110001010110100001000101101000010001101010101111001011101110100100");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "3f9fabb5a161a61565f3c8273");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_60" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_60" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_70
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_11 {new VerilatedContext};
    contextp = contextp_11.get();
    top = new Vtop_module;
a_wide[0] = 0xF51FB3CBu;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x795C2BAFu;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x8BCC8874u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000005u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x664B508Du;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x7FBA2D55u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xDE626F08u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x0000000Du;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x664B508Du;
out_wide[1] = 0x7FBA2D55u;
out_wide[2] = 0xDE626F08u;
out_wide[3] = 0x0000000Du;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_70=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0101100010111100110010001000011101000111100101011100001010111010111111110101000111111011001111001011");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1101110111100110001001101111000010000111111110111010001011010101010101100110010010110101000010001101");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "dde626f087fba2d55664b508d");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_70" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_70" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_80
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_12 {new VerilatedContext};
    contextp = contextp_12.get();
    top = new Vtop_module;
a_wide[0] = 0x912DED58u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x815F3D86u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x112A5C3Du;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000006u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0xA07123D5u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0xEFC5EFD9u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xE6D24BF4u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000004u;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x912DED58u;
out_wide[1] = 0x815F3D86u;
out_wide[2] = 0x112A5C3Du;
out_wide[3] = 0x00000006u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_80=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0110000100010010101001011100001111011000000101011111001111011000011010010001001011011110110101011000");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "0100111001101101001001001011111101001110111111000101111011111101100110100000011100010010001111010101");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "6112a5c3d815f3d86912ded58");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_80" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_80" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_90
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_13 {new VerilatedContext};
    contextp = contextp_13.get();
    top = new Vtop_module;
a_wide[0] = 0x145B4D6Cu;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x1E3CB8D7u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0xE388A85Du;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000000u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0xAAFA9C14u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x10FF8A98u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x493B2ADAu;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000000u;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0xAAFA9C14u;
out_wide[1] = 0x10FF8A98u;
out_wide[2] = 0x493B2ADAu;
out_wide[3] = 0x00000000u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_90=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0000111000111000100010101000010111010001111000111100101110001101011100010100010110110100110101101100");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "0000010010010011101100101010110110100001000011111111100010101001100010101010111110101001110000010100");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "493b2ada10ff8a98aafa9c14");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_90" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_90" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_100
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_14 {new VerilatedContext};
    contextp = contextp_14.get();
    top = new Vtop_module;
a_wide[0] = 0x90622ED4u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x52AA910Du;
 top->a[1]  = a_wide[1];
a_wide[2] = 0xFA4B6F84u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x0000000Eu;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x9697C987u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0xA8547BECu;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x88A2621Fu;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000002u;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x90622ED4u;
out_wide[1] = 0x52AA910Du;
out_wide[2] = 0xFA4B6F84u;
out_wide[3] = 0x0000000Eu;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_100=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "1110111110100100101101101111100001000101001010101010100100010000110110010000011000100010111011010100");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "0010100010001010001001100010000111111010100001010100011110111110110010010110100101111100100110000111");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "efa4b6f8452aa910d90622ed4");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_100" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_100" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_110
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_15 {new VerilatedContext};
    contextp = contextp_15.get();
    top = new Vtop_module;
a_wide[0] = 0x02FFD14Fu;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x99AB1EB0u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x423AA5D7u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000001u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0xE2FEBC14u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x27ED2125u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xA3D8AE6Cu;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x0000000Fu;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0xE2FEBC14u;
out_wide[1] = 0x27ED2125u;
out_wide[2] = 0xA3D8AE6Cu;
out_wide[3] = 0x0000000Fu;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_110=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0001010000100011101010100101110101111001100110101011000111101011000000000010111111111101000101001111");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1111101000111101100010101110011011000010011111101101001000010010010111100010111111101011110000010100");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "fa3d8ae6c27ed2125e2febc14");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_110" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_110" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_120
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_16 {new VerilatedContext};
    contextp = contextp_16.get();
    top = new Vtop_module;
a_wide[0] = 0x197EA1F3u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x32A2D9A2u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x9DFD0A9Du;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000003u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x3FF83F6Fu;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x91FE18F6u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x962B4786u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x0000000Fu;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x197EA1F3u;
out_wide[1] = 0x32A2D9A2u;
out_wide[2] = 0x9DFD0A9Du;
out_wide[3] = 0x00000003u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_120=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0011100111011111110100001010100111010011001010100010110110011010001000011001011111101010000111110011");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1111100101100010101101000111100001101001000111111110000110001111011000111111111110000011111101101111");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "39dfd0a9d32a2d9a2197ea1f3");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_120" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_120" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_130
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_17 {new VerilatedContext};
    contextp = contextp_17.get();
    top = new Vtop_module;
a_wide[0] = 0xA1757248u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0xF847E1DDu;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x1273ACB9u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000009u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0xA79EE752u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x55ABEF0Du;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x7F0C7A1Du;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000001u;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0xA79EE752u;
out_wide[1] = 0x55ABEF0Du;
out_wide[2] = 0x7F0C7A1Du;
out_wide[3] = 0x00000001u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_130=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "1001000100100111001110101100101110011111100001000111111000011101110110100001011101010111001001001000");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "0001011111110000110001111010000111010101010110101011111011110000110110100111100111101110011101010010");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "17f0c7a1d55abef0da79ee752");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_130" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_130" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_140
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_18 {new VerilatedContext};
    contextp = contextp_18.get();
    top = new Vtop_module;
a_wide[0] = 0xDDDE9BF1u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0xA1CCE21Eu;
 top->a[1]  = a_wide[1];
a_wide[2] = 0xD670084Fu;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000001u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x607AED4Eu;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x2A997EC3u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xDB1C0E19u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000009u;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0xDDDE9BF1u;
out_wide[1] = 0xA1CCE21Eu;
out_wide[2] = 0xD670084Fu;
out_wide[3] = 0x00000001u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_140=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0001110101100111000000001000010011111010000111001100111000100001111011011101110111101001101111110001");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1001110110110001110000001110000110010010101010011001011111101100001101100000011110101110110101001110");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "1d670084fa1cce21eddde9bf1");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_140" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_140" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_150
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_19 {new VerilatedContext};
    contextp = contextp_19.get();
    top = new Vtop_module;
a_wide[0] = 0x5BADEBA3u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0xD8C6C076u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0xD282E5DFu;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000002u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x73C61F5Eu;
 top->b[0]  = b_wide[0];
b_wide[1] = 0xACBD67FAu;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x7AE9EDBEu;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x0000000Eu;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x73C61F5Eu;
out_wide[1] = 0xACBD67FAu;
out_wide[2] = 0x7AE9EDBEu;
out_wide[3] = 0x0000000Eu;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_150=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0010110100101000001011100101110111111101100011000110110000000111011001011011101011011110101110100011");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1110011110101110100111101101101111101010110010111101011001111111101001110011110001100001111101011110");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "e7ae9edbeacbd67fa73c61f5e");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_150" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_150" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_160
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_20 {new VerilatedContext};
    contextp = contextp_20.get();
    top = new Vtop_module;
a_wide[0] = 0x292352D0u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0xD1DF2AFCu;
 top->a[1]  = a_wide[1];
a_wide[2] = 0xBEA1569Du;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000009u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x831EC4B0u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x2FFA731Eu;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x22E83AEFu;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x0000000Fu;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x292352D0u;
out_wide[1] = 0xD1DF2AFCu;
out_wide[2] = 0xBEA1569Du;
out_wide[3] = 0x00000009u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_160=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "1001101111101010000101010110100111011101000111011111001010101111110000101001001000110101001011010000");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1111001000101110100000111010111011110010111111111010011100110001111010000011000111101100010010110000");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "9bea1569dd1df2afc292352d0");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_160" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_160" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_170
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_21 {new VerilatedContext};
    contextp = contextp_21.get();
    top = new Vtop_module;
a_wide[0] = 0xD0EE52CAu;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x788EB9C6u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0xCCF7066Au;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000002u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x73701931u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0xD8F212A3u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xE5199C21u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000004u;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x73701931u;
out_wide[1] = 0xD8F212A3u;
out_wide[2] = 0xE5199C21u;
out_wide[3] = 0x00000004u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_170=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "0010110011001111011100000110011010100111100010001110101110011100011011010000111011100101001011001010");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "0100111001010001100110011100001000011101100011110010000100101010001101110011011100000001100100110001");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "4e5199c21d8f212a373701931");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_170" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_170" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_180
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_22 {new VerilatedContext};
    contextp = contextp_22.get();
    top = new Vtop_module;
a_wide[0] = 0x68DC5558u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x52A03E09u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0xA76E4D13u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x00000009u;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x98B91EF1u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0x1602FCAFu;
 top->b[1]  = b_wide[1];
b_wide[2] = 0x808EBCBAu;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x00000005u;
 top->b[3]  = b_wide[3];
    top->sel = 0x0;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x68DC5558u;
out_wide[1] = 0x52A03E09u;
out_wide[2] = 0xA76E4D13u;
out_wide[3] = 0x00000009u;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_180=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "1001101001110110111001001101000100110101001010100000001111100000100101101000110111000101010101011000");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "0101100000001000111010111100101110100001011000000010111111001010111110011000101110010001111011110001");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "0");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "9a76e4d1352a03e0968dc5558");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_180" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_180" << std::endl;
            unpass_total += unpass;
        }
    // Scenario: RandomTest_190
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_23 {new VerilatedContext};
    contextp = contextp_23.get();
    top = new Vtop_module;
a_wide[0] = 0xE8165EC2u;
 top->a[0]  = a_wide[0];
a_wide[1] = 0x0F728C38u;
 top->a[1]  = a_wide[1];
a_wide[2] = 0x5F5DE459u;
 top->a[2]  = a_wide[2];
a_wide[3] = 0x0000000Cu;
 top->a[3]  = a_wide[3];
b_wide[0] = 0x4BFC4499u;
 top->b[0]  = b_wide[0];
b_wide[1] = 0xF686A076u;
 top->b[1]  = b_wide[1];
b_wide[2] = 0xBA1A0667u;
 top->b[2]  = b_wide[2];
b_wide[3] = 0x0000000Du;
 top->b[3]  = b_wide[3];
    top->sel = 0x1;
    top->eval();
    // Checking wide signal out
out_wide[0] = 0x4BFC4499u;
out_wide[1] = 0xF686A076u;
out_wide[2] = 0xBA1A0667u;
out_wide[3] = 0x0000000Du;
    if (top->out != out_wide) {
        unpass++
;printf("===Scenario: RandomTest_190=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "a", "1100010111110101110111100100010110010000111101110010100011000011100011101000000101100101111011000010");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "b", "1101101110100001101000000110011001111111011010000110101000000111011001001011111111000100010010011001");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "sel", "1");

printf("output_vars:\n");
printf("expected %x\n",out_wide[0]);
printf("actual %x\n",top->out[0]);
printf("expected %x\n",out_wide[1]);
printf("actual %x\n",top->out[1]);
printf("expected %x\n",out_wide[2]);
printf("actual %x\n",top->out[2]);
printf("expected %x\n",out_wide[3]);
printf("actual %x\n",top->out[3]);

        printf("Mismatch at %s: expected 0x%s\n", "out", "dba1a0667f686a0764bfc4499");
    }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest_190" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest_190" << std::endl;
            unpass_total += unpass;
        }

    return unpass_total;
}
