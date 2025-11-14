// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtop_module.h for the primary calling header

#include "Vtop_module__pch.h"
#include "Vtop_module__Syms.h"
#include "Vtop_module___024root.h"

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtop_module___024root___dump_triggers__stl(Vtop_module___024root* vlSelf);
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtop_module___024root___eval_triggers__stl(Vtop_module___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop_module__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_module___024root___eval_triggers__stl\n"); );
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VstlTriggered.set(0U, (IData)(vlSelfRef.__VstlFirstIteration));
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtop_module___024root___dump_triggers__stl(vlSelf);
    }
#endif
}

VL_ATTR_COLD void Vtop_module___024root___stl_sequent__TOP__0(Vtop_module___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop_module__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_module___024root___stl_sequent__TOP__0\n"); );
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
    vlSelfRef.done = (3U == (IData)(vlSelfRef.top_module__DOT__state));
    vlSelfRef.out_bytes = ((IData)(vlSelfRef.done) ? vlSelfRef.top_module__DOT__out_bytes_r
                            : 0U);
}

VL_ATTR_COLD void Vtop_module___024root___configure_coverage(Vtop_module___024root* vlSelf, bool first) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtop_module__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtop_module___024root___configure_coverage\n"); );
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    (void)first;  // Prevent unused variable warning
    vlSelf->__vlCoverInsert(&(vlSymsp->__Vcoverage[0]), first, "top_module.v", 16, 9, ".top_module", "v_line/top_module", "case", "16");
    vlSelf->__vlCoverInsert(&(vlSymsp->__Vcoverage[1]), first, "top_module.v", 17, 9, ".top_module", "v_line/top_module", "case", "17");
    vlSelf->__vlCoverInsert(&(vlSymsp->__Vcoverage[2]), first, "top_module.v", 18, 9, ".top_module", "v_line/top_module", "case", "18");
    vlSelf->__vlCoverInsert(&(vlSymsp->__Vcoverage[3]), first, "top_module.v", 19, 8, ".top_module", "v_line/top_module", "case", "19");
    vlSelf->__vlCoverInsert(&(vlSymsp->__Vcoverage[4]), first, "top_module.v", 14, 5, ".top_module", "v_line/top_module", "block", "14-15");
    vlSelf->__vlCoverInsert(&(vlSymsp->__Vcoverage[5]), first, "top_module.v", 24, 3, ".top_module", "v_branch/top_module", "if", "24");
    vlSelf->__vlCoverInsert(&(vlSymsp->__Vcoverage[6]), first, "top_module.v", 24, 4, ".top_module", "v_branch/top_module", "else", "25");
    vlSelf->__vlCoverInsert(&(vlSymsp->__Vcoverage[7]), first, "top_module.v", 23, 5, ".top_module", "v_line/top_module", "block", "23");
    vlSelf->__vlCoverInsert(&(vlSymsp->__Vcoverage[8]), first, "top_module.v", 31, 2, ".top_module", "v_line/top_module", "block", "31-32");
}
