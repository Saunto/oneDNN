/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include <math.h>

#include <random>
#include <sstream>

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "binary/binary.hpp"
#include "reduction/reduction.hpp"

namespace reduction {

// first: nonneutral elements
// second: maximum range
using problem_bounds = std::pair<const int, const int>;

// int | acc | elems | value_range | worst case
//  Y  | mul |  10   |       3     |     3^10=2^16, out of 2^30 (max integer)
//  N  | mul |  30   |       3     | (2^3)^30=2^90, out of 2^128 (max exponent)
//  Y  | sum | 10000 |      50     | 10000*50=2^19, out of 2^30 (max integer)
//  N  | sum | 10000 |      16     | 10000*16=2^18, out of 2^23 (max mantissa/integer)
//  min/max  |  all  |    1000     | no limits on accumulation chain
const problem_bounds MUL_INT = problem_bounds(10, 3);
const problem_bounds MUL_FP = problem_bounds(30, 3);
const problem_bounds SUM_INT = problem_bounds(10000, 50);
const problem_bounds SUM_FP = problem_bounds(10000, 16);
const problem_bounds MINMAX_INT = problem_bounds(-1, 1000);
const problem_bounds MINMAX_FP = problem_bounds(-1, 1000);

problem_bounds get_problem_bounds(alg_t alg, dnnl_data_type_t dt) {
    const bool is_int = is_integral_dt(dt);
    switch (alg) {
        case alg_t::max:
        case alg_t::min: return is_int ? MINMAX_INT : MINMAX_FP;
        case alg_t::mul: return is_int ? MUL_INT : MUL_FP;
        // All remaining cases accumulate via sum
        default: return is_int ? SUM_INT : SUM_FP;
    }
}

dnnl_status_t init_pd(init_pd_args_t<prb_t> &init_pd_args) {
    const prb_t *prb = init_pd_args.prb;

    auto src_desc = dnn_mem_t::init_md(
            prb->ndims, prb->vdims[0].data(), prb->sdt, prb->stag);
    auto dst_desc = dnn_mem_t::init_md(
            prb->ndims, prb->vdims[1].data(), prb->ddt, prb->dtag);

    attr_args_t attr_args;
    attr_args.prepare_post_ops_mds(prb->attr, prb->ndims, prb->vdims[1].data());
    const auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args));

    DNN_SAFE_STATUS(dnnl_reduction_primitive_desc_create(&init_pd_args.pd,
            init_pd_args.engine, alg2alg_kind(prb->alg), src_desc, dst_desc,
            prb->p, prb->eps, dnnl_attr));

    return dnnl_success;
}

bool is_norm_alg(const alg_t alg) {
    return alg == alg_t::norm_lp_max || alg == alg_t::norm_lp_sum
            || alg == alg_t::norm_lp_power_p_max
            || alg == alg_t::norm_lp_power_p_sum;
}

int fill_mem(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        float non_neutral_prob, bool expanded_range,
        bool only_positive_values) {
    const auto sdt = mem_dt.dt();
    const auto ddt = prb->ddt;
    const auto nelems = mem_fp.nelems();
    const float neutral_value = prb->alg == alg_t::mul ? 1.0f : 0.0f;
    // include ddt in is_signed to avoid mistrusted rounding negative -> 0
    const bool is_signed = sdt != dnnl_u8 && ddt != dnnl_u8;
    float shift = 0.0f;
    if (prb->alg == alg_t::mean || (prb->alg == alg_t::min && !is_signed))
        shift = 1.0f;
    const bool is_int = is_integral_dt(sdt);

    // Follow table in comments of fill_src
    int value_range = get_problem_bounds(prb->alg, sdt).second;

    if (expanded_range) value_range = 1000;

    const bool is_mul_fp = prb->alg == alg_t::mul && !is_int;
    const int min_range = is_mul_fp ? -value_range : 1;

    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        const int64_t idx_start = idx_chunk * chunk_size;
        const int64_t idx_end = MIN2(idx_start + chunk_size, nelems);

        std::minstd_rand msr(idx_start + 1);
        msr.discard(1);
        std::uniform_int_distribution<> igen(min_range, value_range);
        std::uniform_int_distribution<> fifty_fifty(0, 1);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value = neutral_value;
            if (flip_coin(idx, non_neutral_prob)) {
                const int gen = igen(msr);
                value = is_mul_fp ? std::pow(2, gen) : gen;

                if (!only_positive_values && is_signed && fifty_fifty(msr) == 1)
                    value = -value;
            }
            value += shift;
            mem_fp.set_elem(idx, round_to_nearest_representable(sdt, value));
        }
    });
    SAFE(mem_dt.reorder(mem_fp), WARN);
    return OK;
}

int fill_src(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    const auto sdt = prb->sdt;
    if (!nelems) return OK;

    int nelems_to_reduce = 1;
    for (int dim = 0; dim < prb->ndims; dim++) {
        if (prb->vdims[0][dim] != prb->vdims[1][dim]) {
            nelems_to_reduce *= prb->vdims[0][dim];
        }
    }

    // Determine number of non-neutral elements to have in the reduction chain
    int safe_to_reduce_elems = get_problem_bounds(prb->alg, sdt).first;
    if (safe_to_reduce_elems == -1) safe_to_reduce_elems = nelems_to_reduce;

    const float non_neutral_prob
            = 1.f * safe_to_reduce_elems / nelems_to_reduce;

    return fill_mem(prb, mem_dt, mem_fp, non_neutral_prob, false, false);
}

int fill_dst(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    const bool only_positive_values = is_norm_alg(prb->alg);
    return fill_mem(prb, mem_dt, mem_fp, 1.0f, true, only_positive_values);
}

void skip_unimplemented_prb(const prb_t *prb, res_t *res) {
    skip_unimplemented_data_type({prb->sdt, prb->ddt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res);
}

void skip_invalid_prb(const prb_t *prb, res_t *res) {
    // Normalization algorithms don't make sense for integer data type.
    // They also can't have `p` parameter less than one.
    const bool is_invalid = is_norm_alg(prb->alg)
            && (is_integral_dt(prb->sdt) || prb->p < 1.f);

    if (is_invalid) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
        return;
    }
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    // accounts for inaccurate rootn/pow functions in norm algs.
    float scale = is_norm_alg(prb->alg) ? 5.0f : 1.0f;
    cmp.set_threshold(scale * epsilon_dt(prb->ddt));
    if (is_amd_gpu()) {
        // MIOpen implementation is less accurate for f16 data type therefore
        // adjust the threshold.
        if (prb->sdt == dnnl_f16 || prb->ddt == dnnl_f16)
            cmp.set_threshold(1.5e-4 * 4);
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prb->ctx_init, prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
    if (is_bench_mode(INIT)) return OK;

    auto const_pd = query_pd(prim);

    const auto fp_dt = dnnl_f32;
    const auto abx_tag = tag::abx;

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();

    const auto &src_md = query_md(const_pd, DNNL_ARG_SRC);
    dnn_mem_t src_fp(src_md, fp_dt, abx_tag, ref_engine);
    dnn_mem_t src_dt(src_md, test_engine);
    SAFE(fill_src(prb, src_dt, src_fp), WARN);

    const auto &dst_md = query_md(const_pd, DNNL_ARG_DST);
    dnn_mem_t dst_fp(dst_md, fp_dt, abx_tag, ref_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0)
        SAFE(fill_dst(prb, dst_dt, dst_fp), WARN);

    const auto sdt = prb->sdt;
    const auto ddt = prb->ddt;
    // include ddt in is_signed to avoid mistrusted rounding negative -> 0
    const bool is_signed = sdt != dnnl_u8 && ddt != dnnl_u8;
    const bool binary_po_only_positive_vals
            = is_norm_alg(prb->alg) || !is_signed;
    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    SAFE(binary::setup_binary_po(const_pd, binary_po_args, binary_po_dt,
                 binary_po_fp, binary_po_only_positive_vals),
            WARN);

    args_t args, ref_args;

    args.set(DNNL_ARG_SRC, src_dt);
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(binary_po_args, binary_po_dt);

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (is_bench_mode(CORR)) {
        ref_args.set(DNNL_ARG_SRC, src_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);
        ref_args.set(binary_po_args, binary_po_fp);

        check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace reduction