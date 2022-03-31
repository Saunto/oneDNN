/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <random>

#include "oneapi/dnnl/dnnl.h"

#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "sum/sum.hpp"

namespace sum {

dnnl_status_t init_pd(dnnl_engine_t engine, const prb_t *prb,
        dnnl_primitive_desc_t &spd, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {
    std::vector<dnnl_memory_desc_t> src_d(prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input)
        src_d[i_input] = dnn_mem_t::init_md(prb->ndims, prb->dims.data(),
                prb->sdt[i_input], prb->stag[i_input]);

    dnnl_memory_desc_t dst_d {};
    if (prb->dtag != tag::undef) {
        dst_d = dnn_mem_t::init_md(
                prb->ndims, prb->dims.data(), prb->ddt, prb->dtag);
    }

    auto dnnl_attr = make_benchdnn_dnnl_wrapper(
            create_dnnl_attr(prb->attr, attr_args_t()));

    return dnnl_sum_primitive_desc_create(&spd,
            prb->dtag != tag::undef ? &dst_d : nullptr, prb->n_inputs(),
            prb->scales.data(), src_d.data(), dnnl_attr, engine);
}

int fill_src(
        const prb_t *prb, int input_idx, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {

    const auto nelems = mem_fp.nelems();
    const auto dt = prb->sdt[input_idx];
    const int range = 16;
    const int f_min = dt == dnnl_u8 ? 0 : -range / 2;

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) - 17 * input_idx + 101) % range;
        const float value = (dt == dnnl_bf16 || dt == dnnl_f16)
                ? (f_min + gen) / range
                : (f_min + gen) * (1.0f + 4.0f / range);
        mem_fp.set_elem(i, round_to_nearest_representable(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    std::vector<dnnl_data_type_t> dts = prb->sdt;
    dts.push_back(prb->ddt);
    check_known_skipped_case_common(dts, prb->dir, res);
}

void setup_cmp(compare::compare_t &cmp, const prb_t *prb, data_kind_t kind,
        const args_t &ref_args) {
    cmp.set_threshold(epsilon_dt(prb->ddt) * prb->n_inputs());
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    check_sum_post_ops(prb->attr, res);
    if (res->state == SKIPPED) return OK;

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);

    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto &test_engine = get_test_engine();
    const auto &ref_engine = get_cpu_engine();
    const auto &dst_md = query_md(const_pd, DNNL_ARG_DST);
    const auto &scratchpad_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);

    dnn_mem_t dst_fp(dst_md, dnnl_f32, tag::abx, ref_engine);
    dnn_mem_t dst_dt(dst_md, test_engine);
    dnn_mem_t scratchpad_dt(scratchpad_md, test_engine);

    args_t args, ref_args;
    args.set(DNNL_ARG_DST, dst_dt);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

    std::vector<dnn_mem_t> src_fp, src_dt;
    src_fp.reserve(prb->n_inputs());
    src_dt.reserve(prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
        const auto &src_md
                = query_md(const_pd, DNNL_ARG_MULTIPLE_SRC + i_input);
        src_fp.emplace_back(src_md, dnnl_f32, tag::abx, ref_engine);
        src_dt.emplace_back(src_md, test_engine);
        SAFE(fill_src(prb, i_input, src_dt[i_input], src_fp[i_input]), WARN);
        args.set(DNNL_ARG_MULTIPLE_SRC + i_input, src_dt[i_input]);
        if (is_bench_mode(CORR))
            ref_args.set(DNNL_ARG_MULTIPLE_SRC + i_input, src_fp[i_input]);
    }

    SAFE(execute_and_wait(prim, args, res), WARN);

    if (is_bench_mode(CORR)) {
        ref_args.set(DNNL_ARG_DST, dst_fp);

        check_correctness(prb, {DST}, args, ref_args, setup_cmp, res);
    }

    return measure_perf(res, prim, args);
}

} // namespace sum
