
#include "rfuzz-harness.h"
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <verilated.h>
#include "Vtop_module.h"

int fuzz_poke() {
    VerilatedContext* contextp;
    Vtop_module* top;
        int unpass_total = 0;
        int unpass = 0;
       ////////////////////////////scenario ResetFunctionality////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_0 {new VerilatedContext};
    contextp = contextp_0.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x1;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "1");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,10, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "1");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,10, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x1;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "1");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,10, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "1");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,10, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x1;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "1");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,10, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "1");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,10, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x1;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "1");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,10, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "1");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,10, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x1;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "1");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,10, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "1");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,10, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,10, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,10, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,10, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,10, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,10, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,10, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,10, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,10, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,10, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ResetFunctionality, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,10, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario ResetFunctionality" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario ResetFunctionality" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario NormalMessage////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_1 {new VerilatedContext};
    contextp = contextp_1.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x81;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x09;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x81;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x09;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x81;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x09;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,15, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,15, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: NormalMessage, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,15, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario NormalMessage" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario NormalMessage" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario ConsecutiveMessages////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_2 {new VerilatedContext};
    contextp = contextp_2.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x81;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x09;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x82;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0c;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x81;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x09;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x82;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0c;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x81;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x09;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x82;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0c;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x00;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: ConsecutiveMessages, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario ConsecutiveMessages" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario ConsecutiveMessages" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario InvalidBytes////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_3 {new VerilatedContext};
    contextp = contextp_3.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x01;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,12, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x02;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,12, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x03;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,12, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x84;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,12, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x05;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,12, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x06;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,12, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x01;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,12, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x02;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,12, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x03;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,12, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x84;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,12, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x05;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,12, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x06;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,12, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: InvalidBytes, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,12, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario InvalidBytes" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario InvalidBytes" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario RandomTest0////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_4 {new VerilatedContext};
    contextp = contextp_4.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe8;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xce;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x3b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x70;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01110000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01110000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x1f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xde;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11011110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11011110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xba;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x53;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01010011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01010011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x90;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x1d;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe6;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe8;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x66;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb1;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xfa;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xfa;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x4b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x4f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01001111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01001111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xbc;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest0, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest0" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest0" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario RandomTest1////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_5 {new VerilatedContext};
    contextp = contextp_5.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xed;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x19;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x45;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x50;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x61;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01100001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01100001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xba;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x5f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe5;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x7b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xff;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xcb;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xf3;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x7b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x43;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x6c;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xfb;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe9;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xaf;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xa4;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10100100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10100100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x33;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest1, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest1" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest1" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario RandomTest2////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_6 {new VerilatedContext};
    contextp = contextp_6.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xc7;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x05;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x8a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe0;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x8f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10001111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10001111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x80;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xdf;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xd0;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xed;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x27;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00100111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00100111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xd7;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x9c;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10011100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10011100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x93;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xf8;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe6;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x9e;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10011110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10011110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x41;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x2f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x93;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xdb;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11011011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest2, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11011011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest2" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest2" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario RandomTest3////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_7 {new VerilatedContext};
    contextp = contextp_7.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb0;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x6f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x11;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00010001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00010001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xec;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x3b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xc2;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x28;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x7f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x73;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xee;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe5;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x33;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xcd;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x8f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10001111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10001111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb2;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe5;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe6;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x75;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x73;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x38;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest3, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest3" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest3" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario RandomTest4////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_8 {new VerilatedContext};
    contextp = contextp_8.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xbd;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x83;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x26;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb2;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x6a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb5;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x9e;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10011110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10011110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x34;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00110100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00110100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x70;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01110000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01110000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xad;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x86;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x67;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01100111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01100111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xae;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xa2;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10100010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10100010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x91;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xdc;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11011100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11011100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe3;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x24;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00100100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00100100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xf5;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x3a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest4, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest4" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest4" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario RandomTest5////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_9 {new VerilatedContext};
    contextp = contextp_9.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xd7;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x5e;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01011110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01011110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x2f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xbf;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xc6;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xef;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xf5;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x4e;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01001110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01001110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xff;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11111111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x2c;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00101100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00101100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x80;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x68;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xca;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x6b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xd0;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb7;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x1a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x1f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x32;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00110010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00110010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb3;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest5, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest5" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest5" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario RandomTest6////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_10 {new VerilatedContext};
    contextp = contextp_10.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x87;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x51;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01010001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01010001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x7c;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x59;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01011001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01011001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x91;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb0;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x7b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xd0;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe0;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xd4;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11010100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xf4;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xeb;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xba;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb7;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xa9;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x9f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x3d;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x26;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x3a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xc6;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest6, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest6" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest6" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario RandomTest7////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_11 {new VerilatedContext};
    contextp = contextp_11.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x85;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x04;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xc2;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xc9;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xf1;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xbc;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x47;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb5;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xbd;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xef;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xae;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x91;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10010001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x7e;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x35;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xed;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xc1;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x1f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x58;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01011000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01011000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x68;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest7, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest7" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest7" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario RandomTest8////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_12 {new VerilatedContext};
    contextp = contextp_12.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xce;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe8;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x43;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x03;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xac;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x80;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xa6;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0d;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0d;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x44;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xa1;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10100001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10100001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x54;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01010100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01010100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x68;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01101000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x50;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01010000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x1c;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x8a;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10001010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x39;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xae;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10101110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x18;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb2;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest8, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110010");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest8" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest8" << std::endl;
            unpass_total += unpass;
        }
       ////////////////////////////scenario RandomTest9////////////////////////////
        unpass = 0;
    const std::unique_ptr<VerilatedContext> contextp_13 {new VerilatedContext};
    contextp = contextp_13.get();
    top = new Vtop_module;
    top->clk = 0;
        top->clk = 1;
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x88;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10001000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 0=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10001000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 0,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb7;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 1=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 1,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xde;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11011110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 2=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11011110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 2,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x83;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 3=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 3,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb1;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 4=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10110001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 4,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xa6;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 5=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10100110");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 5,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xcb;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 6=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 6,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xe0;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 7=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11100000");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 7,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x47;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 8=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01000111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 8,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xc3;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 9=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11000011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 9,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xb9;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 10=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111001");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 10,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x0b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 11=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00001011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 11,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xf5;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 12=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 12,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x1f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 13=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 13,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x3b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 14=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 14,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xf4;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 15=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11110100");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 15,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x7b;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 16=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "01111011");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 16,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xcd;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 17=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "11001101");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 17,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0xbf;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 18=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "10111111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 18,20, "out_bytes", "0x000000");
        }
        contextp->timeInc(1);  
        top->clk = !top->clk;
         top->eval();
         contextp->timeInc(1);
        top->clk = !top->clk;
        top->reset = 0x0;
        top->in = 0x1f;
        top->eval();
        if (top->done != 0x0) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "done", "0x0");
        }
        if (top->out_bytes != 0x000000) {
            unpass++;
printf("===Scenario: RandomTest9, clock cycle: 19=====\n");
printf("input_vars:\n");
printf("top->%s = 0x%s\n", "reset", "0");

printf("input_vars:\n");
printf("top->%s = 0x%s\n", "in", "00011111");

printf("actual %llx\n",top->out_bytes);
            printf("At %d clock cycle of %d, top->%s, expected = 0x%s\n", 19,20, "out_bytes", "0x000000");
        }


        if (unpass == 0) {
            std::cout << "Test passed for scenario RandomTest9" << std::endl;
        } else {
            std::cout << "Test failed,unpass = " << unpass << " for scenario RandomTest9" << std::endl;
            unpass_total += unpass;
        }

    return unpass_total;
}
