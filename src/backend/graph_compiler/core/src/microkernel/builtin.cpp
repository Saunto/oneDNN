/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "builtin.hpp"
#include <array>
#include <tuple>
#include <utility>
#include <compiler/config/context.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <unordered_map>
#include <util/utils.hpp>

SC_MODULE(microkernel.builtin)

using namespace sc::builder;
namespace sc {
namespace builtin {

static sc_data_type_t infer_output_dtype(sc_data_type_t dtype_A) {
    if (dtype_A == datatypes::u8 || dtype_A == datatypes::s8) {
        return datatypes::s32;
    }
    return datatypes::f32;
}

void print_index(expr v) {
    static const func_t print_index_f = make_func("print_index",
            {make_var(datatypes::index, "v")}, stmt(), datatypes::void_t);
    _evaluate_call_(print_index_f, std::move(v));
}

void print_int(expr v) {
    static const func_t print_int_f = make_func("print_int",
            {make_var(datatypes::s32, "v")}, stmt(), datatypes::void_t);
    _evaluate_call_(print_int_f, std::move(v));
}

void print_float(expr v) {
    static const func_t print_float_f = make_func("print_float",
            {make_var(datatypes::f32, "v")}, stmt(), datatypes::void_t);
    _evaluate_call_(print_float_f, std::move(v));
}

void print_str(expr v) {
    static const func_t print_str_f = make_func("print_str",
            {make_var(sc_data_type_t::pointerof(sc_data_etype::U8), "v")},
            stmt(), datatypes::void_t);
    _evaluate_call_(print_str_f, std::move(v));
}

void print_str(const std::string &v) {
    print_str(builder::get_current_builder()->make_str(v));
}

void print_str(const char *v) {
    print_str(std::string(v));
}

expr boundary_check(expr name, expr idx, expr access_len, expr boundary_len) {
    static func_t boundary_check_f = make_func("boundary_check",
            {make_var(datatypes::pointer, "name"),
                    make_var(datatypes::index, "idx"),
                    make_var(datatypes::index, "access_len"),
                    make_var(datatypes::index, "boundary_len")},
            stmt(), datatypes::index);
    return boundary_check_f(std::move(name), std::move(idx),
            std::move(access_len), std::move(boundary_len));
}

expr make_trace(expr func_name, expr in_or_out) {
    static func_t make_trace_f = make_func("sc_make_trace",
            {make_var(datatypes::s32, "func_name"),
                    make_var(datatypes::s32, "in_or_out")},
            stmt(), datatypes::void_t);
    return make_trace_f(std::move(func_name), std::move(in_or_out));
}

expr call_dump_tensor(expr tsr, expr name, expr shape, expr size, expr limit,
        expr outpath, expr format, expr dtype) {
    static func_t dump_tensor_f = make_func("sc_dump_tensor",
            {make_var(datatypes::pointer, "tsr"),
                    make_var(datatypes::pointer, "name"),
                    make_var(datatypes::pointer, "shape"),
                    make_var(datatypes::index, "size"),
                    make_var(datatypes::index, "limit"),
                    make_var(datatypes::pointer, "outpath"),
                    make_var(datatypes::boolean, "format"),
                    make_var(datatypes::index, "dtype")},
            stmt(), datatypes::void_t);
    return dump_tensor_f(std::move(tsr), std::move(name), std::move(shape),
            std::move(size), std::move(limit), std::move(outpath),
            std::move(format), std::move(dtype));
}

expr call_value_check(expr tsr, expr name, expr size) {
    static func_t value_check_f = make_func("sc_value_check",
            {make_var(datatypes::pointer, "tsr"),
                    make_var(datatypes::pointer, "name"),
                    make_var(datatypes::index, "size")},
            stmt(), datatypes::void_t);
    return value_check_f(std::move(tsr), std::move(name), std::move(size));
}

void dnnl_brgemm_init(
        expr C, expr M, expr N, expr LDC, sc_data_type_t dtypeC, expr value) {
    static func_t brgemm_func = _decl_func("dnnl_brgemm_init", datatypes::s32,
            {_arg_("C", datatypes::pointer), _arg_("M", datatypes::s32),
                    _arg_("N", datatypes::s32), _arg_("LDC", datatypes::s32),
                    _arg_("dtypeC", datatypes::s32),
                    _arg_("value", datatypes::f32)});
    _evaluate_call_(brgemm_func, std::move(C), std::move(M), std::move(N),
            std::move(LDC), dtypeC.as_etype_int(), std::move(value));
}

static const char *brgemm_names[] = {
        "dnnl",
        "sc",
};

static const char *get_brgemm_name(scflags_t::brgemm_t backend) {
    return brgemm_names[static_cast<int>(backend)];
}

// returns the kernel creator and kernel caller pair
static std::pair<func_t, func_t> declare_brgemm_kernel_creator(
        scflags_t::brgemm_t backend, brgemm_mode mode) {
    std::stringstream ss;
    if (mode == brgemm_mode::stride) {
        ss << get_brgemm_name(backend) << "_brgemm";
        func_t creator = _decl_func(ss.str() + "_func", datatypes::pointer,
                {_arg_("M", datatypes::s32), _arg_("N", datatypes::s32),
                        _arg_("K", datatypes::s32),
                        _arg_("LDA", datatypes::s32),
                        _arg_("LDB", datatypes::s32),
                        _arg_("LDC", datatypes::s32),
                        _arg_("stride_a", datatypes::s32),
                        _arg_("stride_b", datatypes::s32),
                        _arg_("beta", datatypes::f32),
                        _arg_("dtypeA", datatypes::s32),
                        _arg_("dtypeB", datatypes::s32)});
        func_t caller = _decl_func(ss.str() + "_call", datatypes::void_t,
                {_arg_("func", datatypes::pointer),
                        _arg_("A", datatypes::pointer),
                        _arg_("B", datatypes::pointer),
                        _arg_("C", datatypes::pointer),
                        _arg_("num", datatypes::s32),
                        _arg_("stream", datatypes::pointer)});
        return std::pair<func_t, func_t>(creator, caller);
    } else {
        ss << get_brgemm_name(backend) << "_brgemm_list";
        func_t creator = _decl_func(ss.str() + "_func", datatypes::pointer,
                {_arg_("M", datatypes::s32), _arg_("N", datatypes::s32),
                        _arg_("K", datatypes::s32),
                        _arg_("LDA", datatypes::s32),
                        _arg_("LDB", datatypes::s32),
                        _arg_("LDC", datatypes::s32),
                        _arg_("beta", datatypes::f32),
                        _arg_("dtypeA", datatypes::s32),
                        _arg_("dtypeB", datatypes::s32)});
        func_t caller = _decl_func(ss.str() + "_call", datatypes::void_t,
                {_arg_("func", datatypes::pointer),
                        _arg_("A", datatypes::pointer),
                        _arg_("B", datatypes::pointer),
                        _arg_("C", datatypes::pointer),
                        _arg_("num", datatypes::s32),
                        _arg_("stride_a", datatypes::s32),
                        _arg_("stride_b", datatypes::s32),
                        _arg_("len", datatypes::s32),
                        _arg_("dtypeA", datatypes::s32),
                        _arg_("dtypeB", datatypes::s32),
                        _arg_("stream", datatypes::pointer)});
        return std::pair<func_t, func_t>(creator, caller);
    }
}

std::pair<func_t, func_t> get_brgemm_creator_and_call_func(
        brgemm_mode mode, scflags_t::brgemm_t backend) {
#define DEF_FUNC(back, list_stride) \
    if (mode == brgemm_mode::list_stride \
            && backend == scflags_t::brgemm_t::back) { \
        static std::pair<func_t, func_t> f \
                = declare_brgemm_kernel_creator(backend, mode); \
        return f; \
    }
    // we need a static variable each branch to ensure there will be no
    // duplicated decl for the same func.
    DEF_FUNC(dnnl, stride)
    DEF_FUNC(dnnl, addr_list)
#undef DEF_FUNC
    assert(0 && "Unreachable");
    return std::pair<func_t, func_t>();
}

// returns the kernel creator and kernel caller pair
static func_t declare_brgemm_update_funcs(
        scflags_t::brgemm_t backend, brgemm_mode mode, bool init) {
    std::stringstream ss;
    if (mode == brgemm_mode::stride) {
        ss << get_brgemm_name(backend) << "_brgemm_";
        if (init) { ss << "init_"; }
        ss << "update";
        func_t update = _decl_func(ss.str(), datatypes::s32,
                {_arg_("A", datatypes::pointer), _arg_("B", datatypes::pointer),
                        _arg_("C", datatypes::pointer),
                        _arg_("num", datatypes::s32),
                        _arg_("M", datatypes::s32), _arg_("N", datatypes::s32),
                        _arg_("K", datatypes::s32),
                        _arg_("LDA", datatypes::s32),
                        _arg_("LDB", datatypes::s32),
                        _arg_("LDC", datatypes::s32),
                        _arg_("stride_a", datatypes::s32),
                        _arg_("stride_b", datatypes::s32),
                        _arg_("dtypeA", datatypes::s32),
                        _arg_("dtypeB", datatypes::s32),
                        _arg_("stream", datatypes::pointer)});
        return update;
    } else {
        ss << get_brgemm_name(backend) << "_brgemm_list_update";
        func_t brgemm_func = _decl_func(ss.str(), datatypes::s32,
                {_arg_("A", datatypes::pointer), _arg_("B", datatypes::pointer),
                        _arg_("C", datatypes::pointer),
                        _arg_("num", datatypes::s32),
                        _arg_("M", datatypes::s32), _arg_("N", datatypes::s32),
                        _arg_("K", datatypes::s32),
                        _arg_("LDA", datatypes::s32),
                        _arg_("LDB", datatypes::s32),
                        _arg_("LDC", datatypes::s32),
                        _arg_("stride_a", datatypes::s32),
                        _arg_("stride_b", datatypes::s32),
                        _arg_("len", datatypes::s32),
                        _arg_("dtypeA", datatypes::s32),
                        _arg_("dtypeB", datatypes::s32),
                        _arg_("stream", datatypes::pointer)});
        return brgemm_func;
    }
}

std::pair<func_t, func_t> get_brgemm_update_funcs(
        brgemm_mode mode, scflags_t::brgemm_t backend) {
#define DEF_FUNC(back) \
    if (mode == brgemm_mode::stride && backend == scflags_t::brgemm_t::back) { \
        static std::pair<func_t, func_t> f = std::pair<func_t, func_t>( \
                declare_brgemm_update_funcs(backend, mode, false), \
                declare_brgemm_update_funcs(backend, mode, true)); \
        return f; \
    }
#define DEF_LIST_FUNC(back) \
    if (mode == brgemm_mode::addr_list \
            && backend == scflags_t::brgemm_t::back) { \
        static std::pair<func_t, func_t> f = std::pair<func_t, func_t>( \
                declare_brgemm_update_funcs(backend, mode, false), nullptr); \
        return f; \
    }
    // we need a static variable each branch to ensure there will be no
    // duplicated decl for the same func.
    DEF_FUNC(dnnl)
    DEF_LIST_FUNC(dnnl)
#undef DEF_FUNC
#undef DEF_LIST_FUNC
    assert(0 && "Unreachable");
    return std::pair<func_t, func_t>();
}

void brgemm_init_update(const expr &A, const expr &B, const expr &C,
        const expr &num, const expr &M, const expr &N, const expr &K,
        const expr &LDA, const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, sc_data_type_t dtypeA, sc_data_type_t dtypeB) {
    builder::get_current_builder()->brgemm(A, B, C, num, M, N, K, LDA, LDB, LDC,
            stride_a, stride_b,
            {brgemm_args::cpu_t {true}, dtypeA, dtypeB,
                    infer_output_dtype(dtypeA)});
}

void brgemm_update(const expr &A, const expr &B, const expr &C, const expr &num,
        const expr &M, const expr &N, const expr &K, const expr &LDA,
        const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, sc_data_type_t dtypeA, sc_data_type_t dtypeB) {
    builder::get_current_builder()->brgemm(A, B, C, num, M, N, K, LDA, LDB, LDC,
            stride_a, stride_b,
            {brgemm_args::cpu_t {false}, dtypeA, dtypeB,
                    infer_output_dtype(dtypeA)});
}

void brgemm_list_update(const expr &A, const expr &B, const expr &C,
        const expr &num, const expr &M, const expr &N, const expr &K,
        const expr &LDA, const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, const expr &len, sc_data_type_t dtypeA,
        sc_data_type_t dtypeB) {
    builder::get_current_builder()->list_brgemm(A, B, C, num, M, N, K, LDA, LDB,
            LDC, stride_a, stride_b, len,
            brgemm_args::extra_args_t(brgemm_args::cpu_t {false}, dtypeA,
                    dtypeB, infer_output_dtype(dtypeA)));
}

void brgemm_init(
        expr C, expr M, expr N, expr LDC, sc_data_type_t dtypeC, expr value) {
    dnnl_brgemm_init(std::move(C), std::move(M), std::move(N), std::move(LDC),
            dtypeC, std::move(value));
}

void mem_zero(expr C, const expr &size, sc_data_type_t dtype) {
    static func_t memzerofunc = _decl_func("memset", datatypes::pointer,
            {_arg_("ptr", datatypes::pointer), _arg_("v", datatypes::s32),
                    _arg_("len", datatypes::index)});
    _evaluate_call_(
            memzerofunc, std::move(C), 0, size * utils::get_sizeof_type(dtype));
}

} // namespace builtin
} // namespace sc
