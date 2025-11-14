// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vtop_module__pch.h"

//============================================================
// Constructors

Vtop_module::Vtop_module(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vtop_module__Syms(contextp(), _vcname__, this)}
    , clk{vlSymsp->TOP.clk}
    , in{vlSymsp->TOP.in}
    , reset{vlSymsp->TOP.reset}
    , done{vlSymsp->TOP.done}
    , out_bytes{vlSymsp->TOP.out_bytes}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vtop_module::Vtop_module(const char* _vcname__)
    : Vtop_module(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vtop_module::~Vtop_module() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vtop_module___024root___eval_debug_assertions(Vtop_module___024root* vlSelf);
#endif  // VL_DEBUG
void Vtop_module___024root___eval_static(Vtop_module___024root* vlSelf);
void Vtop_module___024root___eval_initial(Vtop_module___024root* vlSelf);
void Vtop_module___024root___eval_settle(Vtop_module___024root* vlSelf);
void Vtop_module___024root___eval(Vtop_module___024root* vlSelf);

void Vtop_module::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vtop_module::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vtop_module___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        vlSymsp->__Vm_didInit = true;
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vtop_module___024root___eval_static(&(vlSymsp->TOP));
        Vtop_module___024root___eval_initial(&(vlSymsp->TOP));
        Vtop_module___024root___eval_settle(&(vlSymsp->TOP));
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vtop_module___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vtop_module::eventsPending() { return false; }

uint64_t Vtop_module::nextTimeSlot() {
    VL_FATAL_MT(__FILE__, __LINE__, "", "No delays in the design");
    return 0;
}

//============================================================
// Utilities

const char* Vtop_module::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vtop_module___024root___eval_final(Vtop_module___024root* vlSelf);

VL_ATTR_COLD void Vtop_module::final() {
    Vtop_module___024root___eval_final(&(vlSymsp->TOP));
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vtop_module::hierName() const { return vlSymsp->name(); }
const char* Vtop_module::modelName() const { return "Vtop_module"; }
unsigned Vtop_module::threads() const { return 1; }
void Vtop_module::prepareClone() const { contextp()->prepareClone(); }
void Vtop_module::atClone() const {
    contextp()->threadPoolpOnClone();
}
