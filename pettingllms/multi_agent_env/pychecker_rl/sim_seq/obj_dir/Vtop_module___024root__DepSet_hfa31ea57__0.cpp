// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop_module.h for the primary calling header

#include "Vtop_module__pch.h"
#include "Vtop_module__Syms.h"
#include "Vtop_module___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop_module___024root___dump_triggers__ico(Vtop_module___024root* vlSelf);
#endif  // VL_DEBUG

void Vtop_module___024root___eval_triggers__ico(Vtop_module___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop_module__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_module___024root___eval_triggers__ico\n"); );
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VicoTriggered.set(0U, (IData)(vlSelfRef.__VicoFirstIteration));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtop_module___024root___dump_triggers__ico(vlSelf);
    }
#endif
}

VL_INLINE_OPT void Vtop_module___024root___ico_sequent__TOP__0(Vtop_module___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop_module__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_module___024root___ico_sequent__TOP__0\n"); );
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if ((0U == (IData)(vlSelfRef.top_module__DOT__state))) {
        vlSelfRef.top_module__DOT__next = ((8U & (IData)(vlSelfRef.in))
                                            ? 1U : 0U);
        ++(vlSymsp->__Vcoverage[0]);
    } else if ((1U == (IData)(vlSelfRef.top_module__DOT__state))) {
        ++(vlSymsp->__Vcoverage[1]);
        vlSelfRef.top_module__DOT__next = 2U;
    } else if ((2U == (IData)(vlSelfRef.top_module__DOT__state))) {
        ++(vlSymsp->__Vcoverage[2]);
        vlSelfRef.top_module__DOT__next = 3U;
    } else if ((3U == (IData)(vlSelfRef.top_module__DOT__state))) {
        ++(vlSymsp->__Vcoverage[3]);
        vlSelfRef.top_module__DOT__next = ((8U & (IData)(vlSelfRef.in))
                                            ? 1U : 0U);
    }
    ++(vlSymsp->__Vcoverage[4]);
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop_module___024root___dump_triggers__act(Vtop_module___024root* vlSelf);
#endif  // VL_DEBUG

void Vtop_module___024root___eval_triggers__act(Vtop_module___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop_module__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_module___024root___eval_triggers__act\n"); );
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered.set(0U, ((IData)(vlSelfRef.clk) 
                                       & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__clk__0))));
    vlSelfRef.__Vtrigprevexpr___TOP__clk__0 = vlSelfRef.clk;
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtop_module___024root___dump_triggers__act(vlSelf);
    }
#endif
}

VL_INLINE_OPT void Vtop_module___024root___nba_sequent__TOP__0(Vtop_module___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop_module__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_module___024root___nba_sequent__TOP__0\n"); );
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    ++(vlSymsp->__Vcoverage[8]);
    vlSelfRef.top_module__DOT__out_bytes_r = ((0xffff00U 
                                               & (vlSelfRef.top_module__DOT__out_bytes_r 
                                                  << 8U)) 
                                              | (IData)(vlSelfRef.in));
    if (vlSelfRef.reset) {
        ++(vlSymsp->__Vcoverage[5]);
        vlSelfRef.top_module__DOT__state = 0U;
    } else {
        ++(vlSymsp->__Vcoverage[6]);
        vlSelfRef.top_module__DOT__state = vlSelfRef.top_module__DOT__next;
    }
    ++(vlSymsp->__Vcoverage[7]);
    if ((0U == (IData)(vlSelfRef.top_module__DOT__state))) {
        vlSelfRef.top_module__DOT__next = ((8U & (IData)(vlSelfRef.in))
                                            ? 1U : 0U);
        ++(vlSymsp->__Vcoverage[0]);
    } else if ((1U == (IData)(vlSelfRef.top_module__DOT__state))) {
        ++(vlSymsp->__Vcoverage[1]);
        vlSelfRef.top_module__DOT__next = 2U;
    } else if ((2U == (IData)(vlSelfRef.top_module__DOT__state))) {
        ++(vlSymsp->__Vcoverage[2]);
        vlSelfRef.top_module__DOT__next = 3U;
    } else if ((3U == (IData)(vlSelfRef.top_module__DOT__state))) {
        ++(vlSymsp->__Vcoverage[3]);
        vlSelfRef.top_module__DOT__next = ((8U & (IData)(vlSelfRef.in))
                                            ? 1U : 0U);
    }
    ++(vlSymsp->__Vcoverage[4]);
    vlSelfRef.done = (3U == (IData)(vlSelfRef.top_module__DOT__state));
    vlSelfRef.out_bytes = ((IData)(vlSelfRef.done) ? vlSelfRef.top_module__DOT__out_bytes_r
                            : 0U);
}
