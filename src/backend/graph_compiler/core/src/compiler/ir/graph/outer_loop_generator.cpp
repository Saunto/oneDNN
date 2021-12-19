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

#include "outer_loop_generator.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "fusion_mgr.hpp"
#include "utils.hpp"
#include "visitor.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <runtime/config.hpp>
#include <util/utils.hpp>
namespace sc {

SC_MODULE(graph.outer_loop_gen);

static for_loop get_next_inner_loop(const for_loop &cur_loop) {
    if (cur_loop->body_.isa<for_loop>()) {
        return cur_loop->body_.checked_as<for_loop>();
    } else if (cur_loop->body_.isa<stmts>()
            && cur_loop->body_.checked_as<stmts>()->seq_.size() == 1
            && cur_loop->body_.checked_as<stmts>()->seq_[0].isa<for_loop>()) {
        return cur_loop->body_.checked_as<stmts>()
                ->seq_[0]
                .checked_as<for_loop>();
    }
    return for_loop();
}

static int64_t get_loop_range(const for_loop &loop) {
    return (get_const_as_int(loop->iter_end_.checked_as<constant>())
                   - get_const_as_int(loop->iter_begin_.checked_as<constant>()))
            / get_const_as_int(loop->step_.checked_as<constant>());
}

static void fuse_outer_loops(for_loop outer_loop) {
    assert(outer_loop.defined());
    const int max_fused_number
            = runtime_config_t::get().threads_per_instance_ * 10;
    for_loop cur_loop = std::move(outer_loop);
    std::vector<for_loop> loops;
    while (cur_loop.defined()) {
        loops.push_back(cur_loop);
        cur_loop = get_next_inner_loop(cur_loop);
    }
    int64_t fused_number = get_loop_range(loops[0]);
    for (size_t i = 1; i < loops.size() - 1; i++) {
        if (fused_number >= max_fused_number) { break; }
        loops[0]->fuse(loops[i]);
        fused_number = fused_number * get_loop_range(loops[i]);
    }
}

outer_loop_generator_t::outer_loop_generator_t(size_t base_inp_idx)
    : body_generator_base_t({}, {}), base_inp_idx_(base_inp_idx) {}

typedef std::vector<int> (*loop_sort_rule_func)(
        const std::vector<int> &, sc_graph_t &, const tensor &);

static bool axis_can_be_sort(const sc_graph_t &graph) {
    return std::all_of(
            graph.ops_.begin(), graph.ops_.end(), [](const sc_op_ptr &op) {
                return !op->isa<reorder_op_t>() && !op->isa<tensor_view_op_t>();
            });
}

/**
 * Move loop axis of reduce axis to inner.
 *
 * E.g. loop axis is {0, 1, 2, 3}, rd_axis is {1, 2}, after func, we get loop
 * axis {0, 3, 1, 2}
 * */
static std::vector<int> move_reduce_axis_to_inner(
        const std::vector<int> &in_axis, sc_graph_t &graph, const tensor &tsr) {
    if (!axis_can_be_sort(graph)) { return in_axis; }
    std::vector<int> out_axis(in_axis.begin(), in_axis.end());
    op_visitor_t vis = op_visitor_t::dfs_topology_sort(graph.ops_.size());
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        if (auto reduce_node = node->dyn_cast<reduce_op_t>()) {
            auto reduce_axis = reduce_node->get_rd_axis();
            std::sort(reduce_axis.begin(), reduce_axis.end());
            auto shape = reduce_node->get_inputs()[0]
                                 ->details_.get_blocking_dims();
            auto run_threads = runtime_config_t::get().threads_per_instance_;
            /* Due to loop order not only affect outer-loop parallelism,
             * but also inner-loop fusion, which will affect local buffer size(
             * sensitive to cache line size). Further, more performance data
             * maybe required and analyzed to decide which strategy shuold be
             * applied to achieve best performance*/
            // if (parallel_num >= run_threads) { return; }
            for (auto raxis : reduce_axis) {
                auto rend
                        = std::remove(out_axis.begin(), out_axis.end(), raxis);
                assert(rend + 1 == out_axis.end());
                *rend = raxis;
            }
        }
    });
    return out_axis;
}

/**
 * Satisfy continuous access of input tensor include vectorization on last axis
 * and ensure size of each load is more than cache line.
 *
 * E.g. loop axis = {1, 3, 4, 0, 2}
 *
 * IF input tensor(origin shape) is f32(32, 4, 16, 8, 16), last axis is 16
 * which fills up a cache line, after func we get loop axis = {1, 3, 0, 2, 4}.
 * IF input tensor(origin shape) is f32{32, 4, 16, 8, 8}, after func we get loop
 * axis = {1, 0, 2, 3, 4}
 * */
static std::vector<int> continuous_access_satisfaction(
        const std::vector<int> &in_axis, sc_graph_t &graph, const tensor &tsr) {
    assert(in_axis.size() == tsr->dims_.size());
    if (!axis_can_be_sort(graph)) { return in_axis; }
    constexpr int cache_line_size = 64;
    int fill_up_dim = static_cast<int>(tsr->dims_.size()) - 1;
    int dtype_size = utils::get_sizeof_type(tsr->elem_dtype_);
    int cur_load_size = get_expr_as_int(tsr->dims_[fill_up_dim]);
    while (fill_up_dim > 0 && cur_load_size * dtype_size < cache_line_size) {
        fill_up_dim--;
        cur_load_size
                = cur_load_size * get_expr_as_int(tsr->dims_[fill_up_dim]);
    }
    // input tensor is too small that can not fill up a cache line.
    // No need to change loop axis.
    if (fill_up_dim == 0) { return in_axis; }
    std::vector<int> out_axis(in_axis.begin(), in_axis.end());
    for (int i = fill_up_dim; i < static_cast<int>(tsr->dims_.size()); i++) {
        auto rend = std::remove(out_axis.begin(), out_axis.end(), i);
        *rend = i;
    }
    return out_axis;
}

static std::vector<loop_sort_rule_func> loop_sort_rules
        = {move_reduce_axis_to_inner, continuous_access_satisfaction};
bool outer_loop_generator_t::generate(context_ptr ctx, const void *config,
        fusion_manager *fusion, const std::vector<expr> &inputs,
        const std::vector<expr> &outputs, std::vector<for_loop> &loops) const {
    COMPILE_ASSERT(inputs.size() > base_inp_idx_,
            "Expecting at least " << base_inp_idx_ + 1
                                  << " input(s) for outer_loop_generator_t");
    tensor in_tsr = inputs[base_inp_idx_].as<tensor>();
    COMPILE_ASSERT(in_tsr.defined(), "Expecting a tensor");
    auto bld = builder::get_current_builder();
    auto numdims = in_tsr->dims_.size();
    assert(numdims > 0);
    std::vector<expr> loop_vars;
    slice_range cur_tsr_slice;
    std::vector<int> loop_axis;
    loop_vars.reserve(numdims);
    cur_tsr_slice.reserve(numdims);
    loop_axis.reserve(numdims);
    // will create numdims loop vars but uses numdims - 1 because user may sort
    // loop axis
    for (size_t i = 0; i < numdims; i++) {
        bld->push_scope();
        loop_vars.emplace_back(builder::make_var(
                datatypes::index, std::string("__itr_") + std::to_string(i)));
        // outer loops should have tensor slice of length=1
        cur_tsr_slice.emplace_back(std::make_pair(loop_vars.back(), expr(1)));
        loop_axis.push_back(static_cast<int>(i));
    }
    // sort loop axis with rules
    for (auto sort_rule : loop_sort_rules) {
        loop_axis = sort_rule(loop_axis, fusion->get_graph(), in_tsr);
    }
    // generate anchors from inner to outer
    for (size_t i = 0; i < numdims - 1; i++) {
        // loop num is current dimension index
        auto loop_num = loop_axis[numdims - i - 1];
        // upper loop num
        auto upper_loop_num = loop_axis[numdims - i - 2];
        // set full tensor range for loop_num dimension
        cur_tsr_slice[loop_num] = std::make_pair(0, in_tsr->dims_[loop_num]);
        fusion->create_output_fusion_anchor(
                {tensor_slice(in_tsr, slice_range(cur_tsr_slice))});
        auto body = bld->pop_scope();
        auto loop = bld->push_for_loop(loop_vars[upper_loop_num], 0,
                in_tsr->dims_[upper_loop_num], 1, body, true, for_type::NORMAL);
        loops.emplace_back(loop.checked_as<for_loop>());
    }
    cur_tsr_slice[loop_axis[0]]
            = std::make_pair(0, in_tsr->dims_[loop_axis[0]]);
    fusion->create_output_fusion_anchor(
            {tensor_slice(in_tsr, slice_range(cur_tsr_slice))});
    return true;
}

static void schedule_outer_anchor_loops(
        const stmt &body, const for_loop &main_loop) {
    ir_comparer cmper;
    if (body.isa<stmts>()) {
        for (auto &st : body.checked_as<stmts>()->seq_) {
            if (st.isa<for_loop>()) {
                auto cur_for = st.checked_as<for_loop>();
                // if loop is not from outmost anchor
                if (main_loop.defined() && cur_for->equals(main_loop, cmper)) {
                    continue;
                }
                cur_for->kind_ = for_type::PARALLEL;
                auto body = cur_for->body_;
                assert(!body.defined() || body.isa<for_loop>()
                        || body.isa<stmts>());
                // next body has for loop at first
                for_loop next_for;
                while (body.defined()
                        && (body.isa<for_loop>()
                                || (!body.static_as<stmts>()->seq_.empty()
                                        && body.static_as<stmts>()
                                                   ->seq_[0]
                                                   .isa<for_loop>()))) {
                    if (body.isa<for_loop>()) {
                        next_for = body.static_as<for_loop>();
                    } else {
                        next_for = body.static_as<stmts>()
                                           ->seq_[0]
                                           .checked_as<for_loop>();
                    }
                    if (next_for->step_.isa<constant>()
                            && get_expr_as_int(next_for->step_) == 1) {
                        cur_for->fuse(next_for);
                    } else {
                        break;
                    }
                    body = next_for->body_;
                }
                // for input shape with only one dimension.
                if (!main_loop.defined() && !next_for.defined()) {
                    cur_for->kind_ = for_type::NORMAL;
                }
            }
        }
    }
}

void outer_loop_generator_t::schedule_loops(context_ptr ctx, const void *config,
        stmt body, std::vector<for_loop> &fors) const {
    for_loop l0;
    if (!fors.empty()) {
        l0 = fors.back();
        l0->kind_ = for_type::PARALLEL;
        for (auto itr = fors.rbegin() + 1; itr != fors.rend(); ++itr) {
            l0->fuse(*itr);
        }
    }
    // For anchor outside fors
    schedule_outer_anchor_loops(body, l0);
}

bool top_level_anchor_generator_t::generate(context_ptr ctx, const void *config,
        fusion_manager *fusion, const std::vector<expr> &inputs,
        const std::vector<expr> &outputs, std::vector<for_loop> &loops) const {
    slice_range ranges;
    COMPILE_ASSERT(!inputs.empty(),
            "Expecting at least 1 input for top_level_anchor_generator_t");
    tensor in_tsr = inputs[0].as<tensor>();
    COMPILE_ASSERT(in_tsr.defined(), "Expecting a tensor");
    for (auto &dim : in_tsr->dims_) {
        ranges.emplace_back(std::make_pair(0, dim));
    }
    fusion->create_output_fusion_anchor(
            {tensor_slice(inputs[0], std::move(ranges))});
    return true;
}

void top_level_anchor_generator_t::schedule_loops(context_ptr ctx,
        const void *config, stmt body, std::vector<for_loop> &fors) const {
    if (body.isa<stmts>()) {
        auto body_seqs = body.checked_as<stmts>()->seq_;
        for (size_t i = 0; i < body_seqs.size(); i++) {
            if (body_seqs[i].isa<for_loop>()) {
                fuse_outer_loops(body_seqs[i].checked_as<for_loop>());
            }
        }
    } else if (body.isa<for_loop>()) {
        fuse_outer_loops(body.checked_as<for_loop>());
    }
}

ir_module_ptr try_lower_fusion_manager(const context_ptr &ctx,
        outer_loop_generator_t *gen, sc_op *op, fusion_manager *fmgr,
        bool check_parallel, bool just_check,
        std::vector<sc_op_ptr> &out_failed) {
    auto modu = std::make_shared<ir_module_t>(ctx);

    std::vector<expr> ins;
    // real_outs are the output tensors in the function arguments
    std::vector<expr> real_outs;
    auto func = graph::create_func_decl_for_op(op, ins, real_outs);
    // finds if an output can be computed in-place on an "input" of the fusion
    // graph
    auto inplacemap = fmgr->query_inplace();
    // todo: check inplace
    auto main_op_input_size = op->get_inputs().size();
    COMPILE_ASSERT(!op->get_inputs().empty(), "Expecting at least 1 input");
    assert(op->get_inputs().size() == (size_t)fmgr->get_input_op_count());
    assert(op->get_outputs().size() == (size_t)fmgr->get_output_op_count());

    // =======================
    // Start of building function body
    // =======================
    builder::ir_builder_t bld;
    bld.push_scope();

    std::vector<for_loop> loops;
    bool status = gen->generate(ctx, nullptr, fmgr, ins, real_outs, loops);
    assert(status);
    bld.push_returns(true);
    auto body = bld.pop_scope();

    // =======================
    // End of building function body
    // =======================
    // the additional arguments for fmgr, according base_inp_idx_ of gen
    auto base_inp_idx = gen->get_base_inp_idx();
    std::vector<expr> additional_args;
    for (size_t i = 0; i < ins.size(); i++) {
        if (i == base_inp_idx) continue;
        additional_args.emplace_back(ins[i]);
    }
    std::vector<fusion_anchor_data> fuse_state;
    out_failed = fmgr->prepare_and_check(
            ctx, fuse_state, real_outs, additional_args);
    if (!out_failed.empty()) {
        fmgr->clear_anchor();
        return nullptr;
    }
    if (just_check) {
        fmgr->clear_anchor();
        return nullptr;
    }
    fmgr->commit(modu, fuse_state);

    func->body_ = std::move(body);
    gen->schedule_loops(ctx, nullptr, func->body_, loops);
    // check that if we are using the outer most anchor. If so, print a warning.

    if (check_parallel && !loops.empty()) {
        auto l0 = loops.back();
        auto &seq = func->body_.checked_as<stmts>()->seq_;
        for (size_t idx = 0; idx < seq.size(); idx++) {
            if (seq[idx].ptr_same(l0)) {
                if (idx != seq.size() - 2) {
                    SC_MODULE_WARN << "Using non-parallel generator. This may "
                                      "lead to bad performance. Op name="
                                   << op->op_name_;
                }
            }
        }
    }
    modu->add_func({func});
    modu->set_entry_func_idx(0);
    return modu;
}

ir_module_ptr lower_fusion_manager(const context_ptr &ctx,
        outer_loop_generator_t *gen, sc_op *op, fusion_manager *fmgr,
        bool check_parallel) {
    std::vector<sc_op_ptr> out_failed;
    auto ret = try_lower_fusion_manager(
            ctx, gen, op, fmgr, check_parallel, false, out_failed);
    COMPILE_ASSERT(ret, "Fusible Op generation failed");
    return ret;
}

} // namespace sc
