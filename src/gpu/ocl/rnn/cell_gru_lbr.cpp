/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

/*
 * Cell execution GRU with linear before reset
 */

#include "gpu/ocl/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::utils;
using namespace rnn_utils;

template cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru_lbr);
template cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru_lbr);

template <prop_kind_t aprop>
cell_execution_sig((_ref_rnn_common_t<aprop>::cell_execution_gru_lbr)) {
    const conf_t &rnn = this->pd()->rnn_conf;
    const ocl_conf_t &ocl_conf = this->pd()->ocl_conf;

    dim_t cell_wei_iter_offset;

    set_offsets_fwd_gemm(
            rnn, iter, dir, lay, wei_iter_offsets, cell_wei_iter_offset);
    auto cell_layer = !rnn.copy_src_layer && lay == 0
            ? user_data.src_layer(dir, iter)
            : workspace.states(lay - 1, dir, iter);
    auto gemm_cell_layer_fwd = !rnn.copy_src_layer && lay == 0
            ? gemm_layer_fwd_src
            : gemm_layer_fwd;
    auto gemm_diff_wei_cell_layer = !rnn.copy_src_layer && lay == 0
            ? gemm_diff_wei_layer_src
            : gemm_diff_wei_layer;
    auto cell_iter = workspace.states(lay, dir, iter - 1);

    auto scratch_gates_owner = scratch.gates(iter);
    auto scratch_gates = scratch_gates_owner.get();

    if (aprop == prop_kind::forward) {
        // call made when cell execution is enabled
        if (!rnn.merge_gemm_layer)
            CHECK(gemm_primitive(engine, ctx, wei_layer, wei_layer_offset,
                    *cell_layer, 0, *scratch_gates, 0, gemm_cell_layer_fwd));

        CHECK(gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset,
                *cell_iter, 0, *scratch.cell(), 0, gemm_iter_fwd));

        CHECK((this->*elemwise_gru_lbr)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, 1,
                workspace, scratch_gates, nullptr, *scratch.cell(), nullptr,
                nullptr, nullptr, bias, tm_scales, diff_bias));

    } else {
        dim_t cell_diff_wei_iter_off, cell_diff_wei_lay_off;

        set_offsets_bwd_gemm(rnn, iter, dir, lay, cell_diff_wei_iter_off,
                cell_diff_wei_lay_off);

        auto diff_states = scratch.diff_states(lay, dir, 0, iter);
        auto diff_states_iter = scratch.diff_states(lay, dir, 0, iter + 1);
        auto diff_states_layer
                = !rnn.copy_diff_dst_layer && lay + 1 == rnn.n_layer
                ? user_data.diff_dst_layer(dir, iter)
                : scratch.diff_states(lay + 1, dir, rnn.n_states, iter);

        auto diff_states0 = scratch.diff_states(lay, dir, 0, iter);
        auto diff_states1 = scratch.diff_states(lay, dir, rnn.n_states, iter);
        auto diff_gates = scratch.diff_gates(iter);

        CHECK((this->*elemwise_gru_lbr)(ctx, dir, lay, iter, rnn.dhc, rnn.mb,
                ocl_conf.elemwise_bwd_batch_block, workspace, scratch_gates,
                diff_gates.get(), *scratch.cell(), diff_states.get(),
                diff_states_iter.get(), diff_states_layer.get(), bias,
                tm_scales, diff_bias));

        if (!rnn.merge_gemm_layer) {
            CHECK(gemm_primitive(engine, ctx, *diff_gates, 0, *cell_layer, 0,
                    diff_weights_layer, cell_diff_wei_lay_off,
                    gemm_diff_wei_cell_layer));

            CHECK(gemm_primitive(engine, ctx, wei_layer, wei_layer_offset,
                    *diff_gates, 0, *diff_states1, 0, gemm_layer_bwd));
        }

        CHECK(gemm_primitive(engine, ctx, wei_iter, cell_wei_iter_offset,
                *scratch.cell(), 0, *diff_states0, 0, gemm_iter_bwd));

        CHECK(gemm_primitive(engine, ctx, *scratch.cell(), 0, *cell_iter, 0,
                diff_weights_iter, cell_diff_wei_iter_off, gemm_diff_wei_iter));
    }
    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
