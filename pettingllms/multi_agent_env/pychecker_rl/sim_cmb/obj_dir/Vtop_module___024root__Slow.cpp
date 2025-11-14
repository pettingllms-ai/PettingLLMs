// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop_module.h for the primary calling header

#include "Vtop_module__pch.h"
#include "Vtop_module__Syms.h"
#include "Vtop_module___024root.h"

void Vtop_module___024root___ctor_var_reset(Vtop_module___024root* vlSelf);

Vtop_module___024root::Vtop_module___024root(Vtop_module__Syms* symsp, const char* v__name)
    : VerilatedModule{v__name}
    , vlSymsp{symsp}
 {
    // Reset structure values
    Vtop_module___024root___ctor_var_reset(this);
}

void Vtop_module___024root___configure_coverage(Vtop_module___024root* vlSelf, bool first);

void Vtop_module___024root::__Vconfigure(bool first) {
    (void)first;  // Prevent unused variable warning
    Vtop_module___024root___configure_coverage(this, first);
}

Vtop_module___024root::~Vtop_module___024root() {
}

// Coverage
void Vtop_module___024root::__vlCoverInsert(uint32_t* countp, bool enable, const char* filenamep, int lineno, int column,
    const char* hierp, const char* pagep, const char* commentp, const char* linescovp) {
    uint32_t* count32p = countp;
    static uint32_t fake_zero_count = 0;
    std::string fullhier = std::string{VerilatedModule::name()} + hierp;
    if (!fullhier.empty() && fullhier[0] == '.') fullhier = fullhier.substr(1);
    if (!enable) count32p = &fake_zero_count;
    *count32p = 0;
    VL_COVER_INSERT(vlSymsp->_vm_contextp__->coveragep(), VerilatedModule::name(), count32p,  "filename",filenamep,  "lineno",lineno,  "column",column,
        "hier",fullhier,  "page",pagep,  "comment",commentp,  (linescovp[0] ? "linescov" : ""), linescovp);
}
