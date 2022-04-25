/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <unordered_map>

#include "interface/c_types_map.hpp"
#include "interface/value.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/passes/constant_propagation.hpp"
#include "backend/dnnl/passes/memory_planning.hpp"
#include "backend/dnnl/passes/op_executable.hpp"
#include "backend/dnnl/passes/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;
using ltw = impl::logical_tensor_wrapper_t;

struct op_inplace_pair_t {
    op_inplace_pair_t(size_t in_idx, size_t out_idx)
        : in_idx_(in_idx), out_idx_(out_idx) {}
    const size_t in_idx_; // the index, not id
    const size_t out_idx_;
};

std::vector<op_inplace_pair_t> get_op_inplace_pairs(
        op_t &op, fusion_info_mgr_t &mgr) {
    // TODO(xxx) extend the set
    const static std::set<impl::op_kind_t> ops {op_kind::dnnl_mul_scales,
            op_kind::dnnl_add_zps, op_kind::dnnl_reorder, op_kind::dnnl_binary,
            op_kind::dnnl_eltwise, op_kind::dnnl_softmax,
            op_kind::dnnl_logsoftmax, op_kind::dnnl_softmax_bwd,
            op_kind::dnnl_logsoftmax_bwd};
    std::vector<op_inplace_pair_t> pairs;

    // Make post-sum inplace has higher priority since it affects both
    // performance and memory footprint
    if (op.has_attr("fusion_info_key")
            && op.get_attr<int64_t>("fusion_info_key") != -1) {
        // sum post ops support inplace
        int64_t key = op.get_attr<int64_t>("fusion_info_key");
        const auto &pops = mgr.get_info(key).get_post_ops();

        // the post-ops input offset
        size_t index = 1;
        if (op.get_kind() == op_kind::dnnl_convolution
                || op.get_kind() == op_kind::dnnl_matmul
                || op.get_kind() == op_kind::dnnl_convtranspose) {
            index = op.has_attr("with_bias") && op.get_attr<bool>("with_bias")
                    ? 3 // src, wei, bias
                    : 2; // src, wei
        } else if (op.get_kind() == op_kind::dnnl_binary) {
            index = 2;
        } else {
            // do nothing
        }

        std::shared_ptr<value_t> post_sum_input;
        for (int i = 0; i < pops.size(); i++) {
            if (pops[i]->is_post_sum()) {
                post_sum_input = op.get_input_value(index);
                break; // assume only one post sum
            } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_binary) {
                index++;
            } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_eltwise) {
                // FIXME(xx) fused conv may have bias
                index++;
            } else {
                // For eltwise post-ops cases. We just do nothing for such
                // cases.
            }
        }

        if (post_sum_input) {
            auto post_sum_input_lt = post_sum_input->get_logical_tensor();
            auto output_lt = op.get_output_value(0)->get_logical_tensor();
            const bool can_inplace = make_dnnl_memory_desc(post_sum_input_lt)
                    == make_dnnl_memory_desc(output_lt);
            if (can_inplace) { pairs.emplace_back(index, 0); }
        }
    } else if (ops.count(op.get_kind())) {
        auto in0 = op.get_input_value(0)->get_logical_tensor();
        auto out0 = op.get_output_value(0)->get_logical_tensor();
        // always assume in0 and out0 may inplace here, please swap inputs for
        // binary operators to broadcast on src1 and inplace on src0
        const bool can_inplace
                = make_dnnl_memory_desc(in0) == make_dnnl_memory_desc(out0);
        if (can_inplace) { pairs.emplace_back(0, 0); }
    } else if (op.get_kind() == op_kind::dnnl_layernorm_bwd) {
        auto diff_dst = op.get_input_value(1)->get_logical_tensor();
        auto diff_src = op.get_output_value(0)->get_logical_tensor();
        const bool can_inplace = make_dnnl_memory_desc(diff_dst)
                == make_dnnl_memory_desc(diff_src);
        if (can_inplace) { pairs.emplace_back(1, 0); }
    } else {
        // Do nothing
    }

    return pairs;
}

std::shared_ptr<execution_args_set_t> execution_args_set_t::clone() const {
    auto ret = std::make_shared<execution_args_set_t>();

    // clone
    ret->value_mem_map_.reserve(value_mem_map_.size());
    for (auto &val_mem : value_mem_map_) {
        memory cloned_mem(val_mem.second.get_desc(),
                val_mem.second.get_engine(), nullptr);
        ret->value_mem_map_.insert({val_mem.first, cloned_mem});
    }

    auto find_val = [&](const memory &mem) -> value_t * {
        auto pos = std::find_if(value_mem_map_.begin(), value_mem_map_.end(),
                [&](const std::pair<value_t *, memory> &val_mem) {
                    return val_mem.second.get() == mem.get();
                });
        assertm(pos != value_mem_map_.end(), "can't find such mem");
        if (pos != value_mem_map_.end())
            return pos->first;
        else
            return nullptr;
    };

    // copy alias
    ret->mems_use_external_inputs_.reserve(mems_use_external_inputs_.size());
    for (const auto &mem_idx : mems_use_external_inputs_) {
        ret->mems_use_external_inputs_.emplace_back(
                ret->value_mem_map_.at(find_val(mem_idx.first)),
                mem_idx.second);
    }

    ret->mems_use_external_outputs_.reserve(mems_use_external_outputs_.size());
    for (const auto &mem_idx : mems_use_external_outputs_) {
        ret->mems_use_external_outputs_.emplace_back(
                ret->value_mem_map_.at(find_val(mem_idx.first)),
                mem_idx.second);
    }

    ret->mems_use_internal_temporary_.reserve(
            mems_use_internal_temporary_.size());
    for (const auto &mem_offkey : mems_use_internal_temporary_) {
        ret->mems_use_internal_temporary_.emplace_back(
                ret->value_mem_map_.at(find_val(mem_offkey.first)),
                mem_offkey.second);
    }

    ret->mems_use_internal_persistent_.reserve(
            mems_use_internal_persistent_.size());
    for (const auto &mem_offkey : mems_use_internal_persistent_) {
        ret->mems_use_internal_persistent_.emplace_back(
                ret->value_mem_map_.at(find_val(mem_offkey.first)),
                mem_offkey.second);
    }

    ret->topo_ordered_exec_args_.reserve(topo_ordered_exec_args_.size());
    for (const auto &args : topo_ordered_exec_args_) {
        std::unordered_map<int, memory> new_args;
        for (auto &kv : args) {
            int idx = kv.first;
            const memory &mem = kv.second;
            new_args.insert({idx, ret->value_mem_map_.at(find_val(mem))});
        }
        ret->topo_ordered_exec_args_.emplace_back(new_args);
    }

    return ret;
}

void execution_args_set_t::clear() {
    mems_use_external_inputs_.clear();
    mems_use_external_outputs_.clear();
    mems_use_internal_temporary_.clear();
    mems_use_internal_persistent_.clear();
    value_mem_map_.clear();
    topo_ordered_exec_args_.clear();
}

void alias_analyzer_t::clear() {
    alias_map_.clear();
    reverse_alias_map_.clear();
}

impl::status_t alias_analyzer_t::run(std::shared_ptr<subgraph_t> &sg) {
    clear();
    auto &subgraph = sg->get_mutable_ops();
    // find alias values
    for (auto &cur_op : subgraph) {
        if (!is_preprocess_op(*cur_op)) continue;
        value_t *out = cur_op->get_output_value(0).get();
        value_t *in = cur_op->get_input_value(0).get();
        alias_map_.insert({out, in});
        reverse_alias_map_.insert({in, out});
    }
    return impl::status::success;
}

// one input can alias to multiple output
std::vector<const value_t *> alias_analyzer_t::get_alias_outputs(
        const value_t *input) const {
    std::vector<const value_t *> alias_output;
    for (const auto &in_out : reverse_alias_map_) {
        if (in_out.first != input) continue;
        alias_output.emplace_back(in_out.second);
    }
    return alias_output;
}

// a output can alias to only one input
const value_t *alias_analyzer_t::get_alias_input(const value_t *output) const {
    if (alias_map_.count(output)) { return alias_map_.at(output); }
    return nullptr;
}

std::vector<const value_t *> alias_analyzer_t::get_all_aliases(
        const value_t *val) const {
    std::queue<const value_t *> q;
    std::set<const value_t *> visited;

    q.push(val);
    visited.insert(val);
    while (!q.empty()) {
        auto temp = q.front();
        q.pop();
        // visit all alias outputs
        auto alias_outputs = get_alias_outputs(temp);
        for (const auto &alias : alias_outputs) {
            if (visited.count(alias)) continue;
            q.push(alias);
            visited.insert(alias);
        }
        // visit alias input
        auto alias_input = get_alias_input(temp);
        if (alias_input && !visited.count(alias_input)) {
            q.push(alias_input);
            visited.insert(alias_input);
        }
    }

    std::vector<const value_t *> ret;
    ret.reserve(visited.size() - 1);
    for (auto &alias : visited) {
        if (alias == val) continue;
        ret.emplace_back(alias);
    }
    return ret;
}

void memory_planner_t::prepare_args_for_conv_and_matmul(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    exec_args args;

    memory mem;
    size_t index = 0;

    // add input args
    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_SRC, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_WEIGHTS, mem});

    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        exec_args_set_.find_value_mem_map(
                op->get_input_value(index++).get(), mem);
        args.insert({DNNL_ARG_BIAS, mem});
    }

    const fusion_info_t &fusion_info
            = (op->has_attr("fusion_info_key")
                      && op->get_attr<int64_t>("fusion_info_key") != -1)
            ? mgr.get_info(op->get_attr<int64_t>("fusion_info_key"))
            : fusion_info_t();
    const auto &pops = fusion_info.get_post_ops();
    for (int i = 0; i < pops.size(); i++) {
        if (pops[i]->is_post_sum()) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert({DNNL_GRAPH_ARG_POST_SRC, mem});
        } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_binary) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, mem});
        } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_convolution) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert({DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS, mem});
        } else {
        }
    }

    // add output args
    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DST, mem});

    if (op->num_outputs() > 1) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_binary(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    exec_args args;

    memory mem;
    size_t index = 0;

    // add input args
    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_SRC_0, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_SRC_1, mem});

    const fusion_info_t &fusion_info
            = (op->has_attr("fusion_info_key")
                      && op->get_attr<int64_t>("fusion_info_key") != -1)
            ? mgr.get_info(op->get_attr<int64_t>("fusion_info_key"))
            : fusion_info_t();
    const auto &pops = fusion_info.get_post_ops();
    for (int i = 0; i < pops.size(); i++) {
        if (pops[i]->is_post_sum()) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert({DNNL_GRAPH_ARG_POST_SRC, mem});
        } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_binary) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, mem});
        } else {
        }
    }

    // add output args
    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DST, mem});

    if (op->num_outputs() > 1) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_prelu(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    exec_args args;

    memory mem;

    // add input args
    exec_args_set_.find_value_mem_map(op->get_input_value(0).get(), mem);
    args.insert({DNNL_ARG_SRC, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(1).get(), mem);
    args.insert({DNNL_ARG_WEIGHTS, mem});

    // add output args
    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DST, mem});

    if (op->num_outputs() > 1) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_prelu_bwd(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    exec_args args;

    memory mem;

    // add input args
    exec_args_set_.find_value_mem_map(op->get_input_value(0).get(), mem);
    args.insert({DNNL_ARG_SRC, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(1).get(), mem);
    args.insert({DNNL_ARG_WEIGHTS, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(2).get(), mem);
    args.insert({DNNL_ARG_DIFF_DST, mem});

    // add output args
    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DIFF_SRC, mem});

    exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
    args.insert({DNNL_ARG_DIFF_WEIGHTS, mem});

    // scratchpad is always present
    exec_args_set_.find_value_mem_map(op->get_output_value(2).get(), mem);
    args.insert({DNNL_ARG_SCRATCHPAD, mem});

    exec_args_set_.add_exec_args(args);
}

// for single-input-single-output op
void memory_planner_t::prepare_args_for_siso_op(op_t *op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        bool need_scratchpad, bool need_workspace) {
    exec_args args;

    memory mem;
    size_t index = 0;

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_FROM, mem});

    const fusion_info_t &fusion_info
            = (op->has_attr("fusion_info_key")
                      && op->get_attr<int64_t>("fusion_info_key") != -1)
            ? mgr.get_info(op->get_attr<int64_t>("fusion_info_key"))
            : fusion_info_t();
    const auto &pops = fusion_info.get_post_ops();
    for (int i = 0; i < pops.size(); i++) {
        if (pops[i]->is_post_sum()) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert({DNNL_GRAPH_ARG_POST_SRC, mem});
        } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_binary) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, mem});
        } else {
        }
    }

    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_TO, mem});

    if (need_scratchpad && op->num_outputs() > 1) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    if (need_workspace && op->num_outputs() > 2) {
        exec_args_set_.find_value_mem_map(op->get_output_value(2).get(), mem);
        args.insert({DNNL_ARG_WORKSPACE, mem});
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_dnnl_pool(op_t *op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        bool need_scratchpad, bool need_workspace) {
    exec_args args;

    memory mem;
    size_t index = 0;

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_FROM, mem});

    const fusion_info_t &fusion_info
            = (op->has_attr("fusion_info_key")
                      && op->get_attr<int64_t>("fusion_info_key") != -1)
            ? mgr.get_info(op->get_attr<int64_t>("fusion_info_key"))
            : fusion_info_t();
    const auto &pops = fusion_info.get_post_ops();
    for (int i = 0; i < pops.size(); i++) {
        if (pops[i]->is_post_sum()) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert({DNNL_GRAPH_ARG_POST_SRC, mem});
        } else if (pops[i]->get_op()->get_kind() == op_kind::dnnl_binary) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert(
                    {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, mem});
        } else {
        }
    }

    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_TO, mem});

    if (need_scratchpad) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    if (need_workspace) {
        if (need_scratchpad) {
            exec_args_set_.find_value_mem_map(
                    op->get_output_value(2).get(), mem);
            args.insert({DNNL_ARG_WORKSPACE, mem});
        } else {
            exec_args_set_.find_value_mem_map(
                    op->get_output_value(1).get(), mem);
            args.insert({DNNL_ARG_WORKSPACE, mem});
        }
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_pool_bwd(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    exec_args args;

    memory mem;

    exec_args_set_.find_value_mem_map(op->get_input_value(0).get(), mem);
    args.insert({DNNL_ARG_DIFF_DST, mem});

    if (op->get_attr<std::string>("kind") == "maxpool") {
        // maxpool bwd op must need workspace input
        exec_args_set_.find_value_mem_map(op->get_input_value(1).get(), mem);
        args.insert({DNNL_ARG_WORKSPACE, mem});
    }

    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DIFF_SRC, mem});

    exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
    args.insert({DNNL_ARG_SCRATCHPAD, mem});

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_miso_op(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    exec_args args;
    memory mem;

    for (int i = 0; i < op->num_inputs(); ++i) {
        exec_args_set_.find_value_mem_map(
                op->get_input_value(static_cast<size_t>(i)).get(), mem);
        args.insert({DNNL_ARG_MULTIPLE_SRC + i, mem});
    }

    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DST, mem});

    if (op->num_outputs() > 1) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_niso_op(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    exec_args args;
    memory mem;

    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    // We only set dst argument, to which constant data will be copied
    args.insert({DNNL_ARG_TO, mem});

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::bind_memory_for_bn_folding(
        op_t *op, const dnnl::engine &p_engine) {
    exec_args args;
    memory mem;

    bool with_bias = op->get_attr<bool>("with_bias");

#define INSERT_ARGS(key, val_offset, direction) \
    exec_args_set_.find_value_mem_map( \
            op->get_##direction##_value(val_offset).get(), mem); \
    args.insert({key, mem});

    // bind input memory
    size_t in_idx = 0;
    INSERT_ARGS(DNNL_ARG_WEIGHTS, in_idx++, input); // weight
    if (with_bias) {
        INSERT_ARGS(DNNL_ARG_BIAS, in_idx++, input); // bias
    }
    INSERT_ARGS(DNNL_ARG_WEIGHTS_1, in_idx++, input); // scale
    INSERT_ARGS(DNNL_ARG_WEIGHTS_2, in_idx++, input); // shift
    INSERT_ARGS(DNNL_ARG_MEAN, in_idx++, input); // mean
    INSERT_ARGS(DNNL_ARG_VARIANCE, in_idx++, input); // variance

    // bind output memory
    size_t out_idx = 0;
    INSERT_ARGS(DNNL_ARG_DST_0, out_idx++, output); // updated weight
    INSERT_ARGS(DNNL_ARG_DST_1, out_idx++, output); // updated bias
    INSERT_ARGS(DNNL_ARG_SCRATCHPAD, out_idx++, output); // scratchpad

#undef INSERT_ARGS
    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::bind_memory_for_conv_bwd_data(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    memory mem;
    size_t index = 0;
    exec_args args;

    // bind mem for inputs
    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_DIFF_DST, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_WEIGHTS, mem});

    // bind mem for outputs
    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DIFF_SRC, mem});

    if (op->num_outputs() > 1) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::bind_memory_for_conv_bwd_weights(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    memory mem;
    size_t index = 0;
    exec_args args;

    // bind mem for inputs
    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_SRC, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_DIFF_DST, mem});

    // bind mem for outputs
    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DIFF_WEIGHTS, mem});

    if (op->num_outputs() > 1) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::bind_memory_for_batchnorm(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    memory mem;
    exec_args args;

    size_t in_index = 0;
#define INSERT_ARGS(key, val_offset, direction) \
    exec_args_set_.find_value_mem_map( \
            op->get_##direction##_value(val_offset).get(), mem); \
    args.insert({key, mem});

    // bind mem for inputs
    INSERT_ARGS(DNNL_ARG_SRC, in_index++, input);
    if (!op->get_attr<bool>("is_training")) { // inference
        INSERT_ARGS(DNNL_ARG_SCALE, in_index++, input);
        INSERT_ARGS(DNNL_ARG_SHIFT, in_index++, input);
        INSERT_ARGS(DNNL_ARG_MEAN, in_index++, input);
        INSERT_ARGS(DNNL_ARG_VARIANCE, in_index++, input);
    } else { // training
        if (op->num_inputs() > 3) {
            INSERT_ARGS(DNNL_ARG_SCALE, in_index++, input);
            INSERT_ARGS(DNNL_ARG_SHIFT, in_index++, input);
        }
        // running_mean/running_variance of last iteration
        INSERT_ARGS(DNNL_ARG_SRC_1, in_index++, input);
        INSERT_ARGS(DNNL_ARG_SRC_2, in_index++, input);
    }

    size_t out_index = 0;
    // bind mem for outputs
    INSERT_ARGS(DNNL_ARG_DST, out_index++, output);
    if (op->get_attr<bool>("is_training")) {
        // running_mean
        INSERT_ARGS(DNNL_ARG_DST_1, out_index++, output);
        // running_variance
        INSERT_ARGS(DNNL_ARG_DST_2, out_index++, output);
        // batch_meam
        INSERT_ARGS(DNNL_ARG_MEAN, out_index++, output);
        // batch_variance
        INSERT_ARGS(DNNL_ARG_VARIANCE, out_index++, output);
    }

    // scratchpad
    if (op->num_outputs() > out_index) {
        INSERT_ARGS(DNNL_ARG_SCRATCHPAD, out_index++, output);
    }

    // workspace (for BatchNormForwardTraining with ReLU)
    if (op->num_outputs() > out_index + 1) {
        INSERT_ARGS(DNNL_ARG_WORKSPACE, out_index++, output);
    }

#undef INSERT_ARGS
    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::bind_memory_for_batchnorm_bwd(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    memory mem;
    size_t index = 0;
    exec_args args;

    // bind mem for inputs
    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_SRC, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_DIFF_DST, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_SCALE, mem});
    args.insert({DNNL_ARG_SHIFT, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_MEAN, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_VARIANCE, mem});

    // bind mem for outputs
    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DIFF_SRC, mem});

    exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
    args.insert({DNNL_ARG_DIFF_SCALE, mem});

    exec_args_set_.find_value_mem_map(op->get_output_value(2).get(), mem);
    args.insert({DNNL_ARG_DIFF_SHIFT, mem});

    if (op->num_outputs() > 3) {
        exec_args_set_.find_value_mem_map(op->get_output_value(3).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::bind_memory_for_layernorm(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    memory mem;
    exec_args args;

    size_t in_index = 0;
#define INSERT_ARGS(key, val_offset, direction) \
    exec_args_set_.find_value_mem_map( \
            op->get_##direction##_value(val_offset).get(), mem); \
    args.insert({key, mem});

    // bind mem for inputs
    INSERT_ARGS(DNNL_ARG_SRC, in_index++, input);
    if (!op->has_attr("use_affine") || op->get_attr<bool>("use_affine")) {
        INSERT_ARGS(DNNL_ARG_SCALE, in_index++, input);
        INSERT_ARGS(DNNL_ARG_SHIFT, in_index++, input);
    }

    size_t out_index = 0;
    // bind mem for outputs
    INSERT_ARGS(DNNL_ARG_DST, out_index++, output);
    if (!op->has_attr("keep_stats") || op->get_attr<bool>("keep_stats")) {
        // meam
        INSERT_ARGS(DNNL_ARG_MEAN, out_index++, output);
        // variance
        INSERT_ARGS(DNNL_ARG_VARIANCE, out_index++, output);
    }

    // scratchpad
    if (op->num_outputs() > out_index) {
        INSERT_ARGS(DNNL_ARG_SCRATCHPAD, out_index++, output);
    }

#undef INSERT_ARGS
    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::bind_memory_for_layernorm_bwd(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    memory mem;
    exec_args args;

    size_t in_index {0};
    exec_args_set_.find_value_mem_map(
            op->get_input_value(in_index++).get(), mem);
    args.insert({DNNL_ARG_SRC, mem});

    exec_args_set_.find_value_mem_map(
            op->get_input_value(in_index++).get(), mem);
    args.insert({DNNL_ARG_DIFF_DST, mem});

    exec_args_set_.find_value_mem_map(
            op->get_input_value(in_index++).get(), mem);
    args.insert({DNNL_ARG_MEAN, mem});

    exec_args_set_.find_value_mem_map(
            op->get_input_value(in_index++).get(), mem);
    args.insert({DNNL_ARG_VARIANCE, mem});

    size_t out_index {0};
    exec_args_set_.find_value_mem_map(
            op->get_output_value(out_index++).get(), mem);
    args.insert({DNNL_ARG_DIFF_SRC, mem});

    if (op->get_attr<bool>("with_gamma")) {
        exec_args_set_.find_value_mem_map(
                op->get_input_value(in_index++).get(), mem);
        args.insert({DNNL_ARG_SCALE, mem});

        exec_args_set_.find_value_mem_map(
                op->get_output_value(out_index++).get(), mem);
        args.insert({DNNL_ARG_DIFF_SCALE, mem});
    }
    if (op->get_attr<bool>("with_beta")) {
        exec_args_set_.find_value_mem_map(
                op->get_input_value(in_index++).get(), mem);
        args.insert({DNNL_ARG_SHIFT, mem});

        exec_args_set_.find_value_mem_map(
                op->get_output_value(out_index++).get(), mem);
        args.insert({DNNL_ARG_DIFF_SHIFT, mem});
    }

    if (op->num_outputs() > out_index) {
        exec_args_set_.find_value_mem_map(
                op->get_output_value(out_index).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_reorder_op(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    exec_args args;

    memory mem;
    size_t index = 0;

    // src
    exec_args_set_.find_value_mem_map(op->get_input_value(index++).get(), mem);
    args.insert({DNNL_ARG_FROM, mem});

    // we always insert the input belonging to input fusion before the input
    // belonging to output fusion. So, src_zps must be before scales if it
    // exists
    if (op->has_attr("with_runtime_src_zps")
            && op->get_attr<bool>("with_runtime_src_zps")) {
        auto src_zps = op->get_input_value(index++);
        assertm(src_zps->get_logical_tensor().data_type == impl::data_type::s32,
                "oneDNN runtime zps must be s32 type");
        exec_args_set_.find_value_mem_map(src_zps.get(), mem);
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, mem});
    }

    if (op->has_attr("with_runtime_scales")
            && op->get_attr<bool>("with_runtime_scales")) {
        exec_args_set_.find_value_mem_map(
                op->get_input_value(index++).get(), mem);
        args.insert({DNNL_ARG_ATTR_OUTPUT_SCALES, mem});
    }

    if (op->has_attr("with_runtime_dst_zps")
            && op->get_attr<bool>("with_runtime_dst_zps")) {
        auto dst_zps = op->get_input_value(index++);
        assertm(dst_zps->get_logical_tensor().data_type == impl::data_type::s32,
                "oneDNN runtime zps must be s32 type");
        exec_args_set_.find_value_mem_map(dst_zps.get(), mem);
        args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, mem});
    }

    const fusion_info_t &fusion_info
            = (op->has_attr("fusion_info_key")
                      && op->get_attr<int64_t>("fusion_info_key") != -1)
            ? mgr.get_info(op->get_attr<int64_t>("fusion_info_key"))
            : fusion_info_t();
    const auto &pops = fusion_info.get_post_ops();
    for (int i = 0; i < pops.size(); i++) {
        if (pops[i]->is_post_sum()) {
            exec_args_set_.find_value_mem_map(
                    op->get_input_value(index++).get(), mem);
            args.insert({DNNL_GRAPH_ARG_POST_SRC, mem});
        } else {
            assertm(false, "oneDNN reorder only support sum post-ops");
        }
    }

    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_TO, mem});

    if (op->num_outputs() == 2) {
        exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
        args.insert({DNNL_ARG_SCRATCHPAD, mem});
    }

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_softmax_bwd(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    memory mem;
    exec_args args;

    // bind mem for inputs
    exec_args_set_.find_value_mem_map(op->get_input_value(0).get(), mem);
    args.insert({DNNL_ARG_DIFF_DST, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(1).get(), mem);
    args.insert({DNNL_ARG_DST, mem});

    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DIFF_SRC, mem});

    exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
    args.insert({DNNL_ARG_SCRATCHPAD, mem});

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_resampling_bwd(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    memory mem;
    exec_args args;

    exec_args_set_.find_value_mem_map(op->get_input_value(1).get(), mem);
    args.insert({DNNL_ARG_DIFF_DST, mem});

    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DIFF_SRC, mem});

    // scratchpad is always present
    exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
    args.insert({DNNL_ARG_SCRATCHPAD, mem});

    exec_args_set_.add_exec_args(args);
}

void memory_planner_t::prepare_args_for_eltwise_bwd(
        op_t *op, const dnnl::engine &p_engine, fusion_info_mgr_t &mgr) {
    UNUSED(mgr);
    memory mem;
    exec_args args;

    exec_args_set_.find_value_mem_map(op->get_input_value(0).get(), mem);
    args.insert(
            {op->get_attr<bool>("use_dst") ? DNNL_ARG_DST : DNNL_ARG_SRC, mem});

    exec_args_set_.find_value_mem_map(op->get_input_value(1).get(), mem);
    args.insert({DNNL_ARG_DIFF_DST, mem});

    exec_args_set_.find_value_mem_map(op->get_output_value(0).get(), mem);
    args.insert({DNNL_ARG_DIFF_SRC, mem});

    // scratchpad is always present
    exec_args_set_.find_value_mem_map(op->get_output_value(1).get(), mem);
    args.insert({DNNL_ARG_SCRATCHPAD, mem});

    exec_args_set_.add_exec_args(args);
}

// Assign partition's input edges to user given external inputs buffer. Those
// external inputs buffers may be used by other partition (which is under the
// control of user), so we can't reuse them.
// Note: Because those external inputs buffers may be used by preprocess op, so
// we also find the edges that share the same buffers and assign the same buffer
// to them.
impl::status_t memory_planner_t::assign_external_inputs_buffer(
        const std::vector<op_ptr> &subgraph,
        const std::vector<impl::logical_tensor_t> &inputs) {
    // Remove duplicated input values
    auto sg_ins = impl::graph_t(subgraph).get_input_values();
    std::sort(sg_ins.begin(), sg_ins.end());
    sg_ins.erase(std::unique(sg_ins.begin(), sg_ins.end()), sg_ins.end());

    // Assign external input buffer to subgraph's inputs and their alias
    for (auto &val : sg_ins) {
        for (size_t i = 0; i < inputs.size(); i++) {
            if (val->get_logical_tensor().id == inputs[i].id) {
                assign_info_t info(external_input, i);
                buffer_assignments_.insert(std::make_pair(val, info));
                // assign alias
                auto aliases = alias_analyzer_.get_all_aliases(val);
                for (auto &alias : aliases) {
                    assertm(!buffer_assignments_.count(alias),
                            "alias of input has been assigned buffer");
                    buffer_assignments_.insert(std::make_pair(alias, info));
                }
                break;
            }
        }
    }

    // Get the live range of external inputs
    size_t time_point = 0;
    impl::topo_order_visit(
            impl::graph_t(subgraph).get_output_ops(), [&](op_t *op) {
                auto in_vals = op->get_input_values();
                for (auto &in_val : in_vals) {
                    if (!buffer_assignments_.count(in_val.get())) continue;
                    const auto &info = buffer_assignments_.at(in_val.get());
                    if (info.kind_ != external_input) continue;
                    external_inputs_live_range_[&info]
                            = time_bound_t {0, time_point};
                }
                time_point++;
                return impl::status::success;
            });

    return status::success;
}

// Assign partition's output edges to user given external outputs buffer. Those
// external outputs buffers may contain valid content (for example the inplace
// scenarios, partition's output share same buffer with inputs. This is under
// the control of user, the library can't know this in compilation), so we can't
// reuse them.
// Note: Because those external outputs buffers may be used by preprocess op, so
// we also find the edges that share the same buffers and assign the same buffer
// to them.
impl::status_t memory_planner_t::assign_external_outputs_buffer(
        const std::vector<op_ptr> &subgraph,
        const std::vector<impl::logical_tensor_t> &outputs,
        fusion_info_mgr_t &mgr) {
    for (auto &val : impl::graph_t(subgraph).get_output_values()) {
        for (size_t i = 0; i < outputs.size(); i++) {
            if (val->get_logical_tensor().id == outputs[i].id) {
                assign_info_t orig_info = buffer_assignments_.at(val);
                assign_info_t updated_info(external_output, i);
                std::queue<const value_t *> q;
                std::set<const value_t *> visited;
                q.push(val);
                while (!q.empty()) {
                    auto cur_val = q.front();
                    q.pop();
                    if (visited.count(cur_val)) continue;

                    // update the assigned buffer to external buffer
                    buffer_assignments_[cur_val] = updated_info;
                    visited.insert(cur_val);

                    // push the alias to queue for next visit
                    auto aliases = alias_analyzer_.get_all_aliases(cur_val);
                    for (const value_t *alias : aliases) {
                        q.push(alias);
                    }

                    // push the inplaced input to queue for next visit
                    auto &producer = cur_val->get_producer();
                    auto op_inplace_pairs = get_op_inplace_pairs(producer, mgr);
                    for (auto &pair : op_inplace_pairs) {
                        if (pair.out_idx_ != cur_val->get_offset()) continue;
                        auto in_val = producer.get_input_value(pair.in_idx_);
                        if (buffer_assignments_.at(in_val.get()) != orig_info)
                            continue;
                        q.push(in_val.get());
                    }
                }
            }
        }
    }
    return status::success;
}

// Assign internal constant edges (such as the const weight reorder's output) to
// persistent buffer. Those persistent buffers will be cached to the global
// constant cache, so they can't be reused anymore.
// Note: Not all constant edges' buffer should be cached. We will find the final
// output edges of the constant block (a block of ops who output constant
// tensor), and only cache the constant block's outputs' buffer. Because those
// outputs may be produced by inplace op, so we also find the edges that share
// the same buffers and assign the same buffer to them. This can be regarded as
// a kind of constant folding, with which the cached buffer can be reduced.
impl::status_t memory_planner_t::assign_internal_persistent_buffer(
        const std::vector<op_ptr> &subgraph, fusion_info_mgr_t &mgr) {
    for (auto &val : get_constant_block_output_values(subgraph)) {
        assign_info_t orig_info = buffer_assignments_.at(val);
        if (orig_info.kind_ != internal_temporary) continue;

        size_t idx = persistent_buffer_assigner_.request(
                make_dnnl_memory_desc(val->get_logical_tensor()).get_size());
        assign_info_t updated_info(internal_persistent, idx);
        std::queue<const value_t *> q;
        std::set<const value_t *> visited;
        q.push(val);
        while (!q.empty()) {
            auto cur_val = q.front();
            q.pop();
            if (visited.count(cur_val) || !cur_val->has_producer()) continue;

            // update the assigned buffer to external buffer
            buffer_assignments_[cur_val] = updated_info;
            visited.insert(cur_val);

            // push the alias to queue for next visit
            auto aliases = alias_analyzer_.get_all_aliases(cur_val);
            for (const value_t *alias : aliases) {
                q.push(alias);
            }

            // push the inplaced input to queue for next visit
            auto &producer = cur_val->get_producer();
            auto op_inplace_pairs = get_op_inplace_pairs(producer, mgr);
            for (auto &pair : op_inplace_pairs) {
                if (pair.out_idx_ != cur_val->get_offset()) continue;
                auto in_val = producer.get_input_value(pair.in_idx_);
                if (buffer_assignments_.at(in_val.get()) != orig_info) continue;
                q.push(in_val.get());
            }
        }
    }
    return status::success;
}

// Assign internal non constant edges (such as src reorder output in conv
// pattern) to temporary buffer. Those temporary buffer will be dynamically
// allocated/freed during execution. In order to reduce memory footprint, we
// introduce two kind of memory optimization:
// - Inplace:  if the op support inplace computation, the output results can be
//   written into input buffer
// - Standard Memory Sharing: if a edge's all consumers have been computed, then
//   the buffer of this edge can be reused by other edge.
// TODO(qun) Consider more situations (for example, a tensor can also be reused
// even if its consumer is not computed, as long as it consumer only need the
// tensor's metadata instead of content)
impl::status_t memory_planner_t::assign_internal_temporary_buffer(
        const std::vector<op_ptr> &subgraph,
        const std::unordered_map<value_t *, size_t> &edge_ref_count,
        fusion_info_mgr_t &mgr, bool enable_standard_sharing) {
    auto func = [&](impl::op_t *op) {
        // Handle alias first
        auto inputs = op->get_input_values();
        for (auto &in : inputs) {
            auto alias_outputs = alias_analyzer_.get_alias_outputs(in.get());
            for (auto &alias : alias_outputs) {
                if (buffer_assignments_.count(alias)) { continue; }
                assign_info_t info = buffer_assignments_.at(in.get());
                buffer_assignments_.insert(std::make_pair(alias, info));
                temporary_buffer_ref_count_[info.index_]
                        += edge_ref_count.at(const_cast<value_t *>(alias));
            }
        }

        // Handle inplace
        auto op_inplace_pairs = get_op_inplace_pairs(*op, mgr);
        if (!op_inplace_pairs.empty()) {
            for (const auto &pair : op_inplace_pairs) {
                value_t *in = op->get_input_value(pair.in_idx_).get();
                assign_info_t info = buffer_assignments_.at(in);
                if (info.kind_ != internal_temporary) continue;

                bool reuse_in_buffer
                        = temporary_buffer_ref_count_[info.index_] == 1;
                if (reuse_in_buffer) {
                    value_t *out = op->get_output_value(pair.out_idx_).get();
                    if (!buffer_assignments_.count(out)) {
                        buffer_assignments_.insert(std::make_pair(out, info));
                        temporary_buffer_ref_count_[info.index_]
                                += edge_ref_count.at(out);
                    }
                }
            }
        }

        // Allocate outputs
        for (auto &out : op->get_output_values()) {
            // already assigned buffer, skip it
            if (buffer_assignments_.count(out.get())) continue;

            // this output need a new buffer, record it
            auto lt = out->get_logical_tensor();
            size_t idx = temporary_buffer_assigner_.request(
                    make_dnnl_memory_desc(lt).get_size());
            buffer_assignments_.insert(std::make_pair(
                    out.get(), assign_info_t(internal_temporary, idx)));
            temporary_buffer_ref_count_[idx] = edge_ref_count.at(out.get());
        }

        // Free inputs
        for (auto &in : op->get_input_values()) {
            assign_info_t info = buffer_assignments_.at(in.get());
            if (info.kind_ != internal_temporary) continue;

            --temporary_buffer_ref_count_[info.index_];
            // if we decrease it to zero, we are ready to release
            if (enable_standard_sharing
                    && temporary_buffer_ref_count_[info.index_] == 0) {
                temporary_buffer_assigner_.release(info.index_);
            }
        }

        // Free outputs that have no consumer (such as scratchpad)
        for (auto &out : op->get_output_values()) {
            assign_info_t info = buffer_assignments_.at(out.get());
            if (info.kind_ != internal_temporary) continue;

            auto consumers = out->get_consumers();
            if (consumers.empty()) {
                --temporary_buffer_ref_count_[info.index_];
                if (enable_standard_sharing) {
                    temporary_buffer_assigner_.release(info.index_);
                }
            }
        }

        return impl::status::success;
    };

    return impl::topo_order_visit(
            impl::graph_t(subgraph).get_output_ops(), func);
}

impl::status_t memory_planner_t::prepare_execution_args_set(
        const std::vector<op_ptr> &subgraph, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr) {
    // bind memory object to each value
    for (value_t *in : impl::graph_t(subgraph).get_input_values()) {
        exec_args_set_.add_value_mem_map({in,
                make_dnnl_memory(
                        make_dnnl_memory_desc(in->get_logical_tensor()),
                        p_engine, nullptr)});
    }

    status_t ret = impl::topo_order_visit(
            impl::graph_t(subgraph).get_output_ops(), [&](impl::op_t *op) {
                for (auto &out : op->get_output_values()) {
                    exec_args_set_.add_value_mem_map({out.get(),
                            make_dnnl_memory(make_dnnl_memory_desc(
                                                     out->get_logical_tensor()),
                                    p_engine, nullptr)});
                }
                return impl::status::success;
            });
    if (ret != status::success) return ret;

    registrar_t temporary_registrar = temporary_registry_.registrar();
    registrar_t persistent_registrar = persistent_registry_.registrar();

    // classify binded memory objects and their index to buffer
    for (const auto &it : exec_args_set_.get_value_mem_map()) {
        value_t *val = it.first;
        const dnnl::memory &mem = it.second;
        const assign_info_t &info = buffer_assignments_.at(val);
        switch (info.kind_) {
            case external_input:
                exec_args_set_.add_mem_use_external_inputs({mem, info.index_});
                break;
            case external_output:
                exec_args_set_.add_mem_use_external_outputs({mem, info.index_});
                break;
            case internal_temporary:
                temporary_registrar.book(info.index_,
                        temporary_buffer_assigner_.query_size(info.index_));
                exec_args_set_.add_mem_use_internal_temporary(
                        {mem, info.index_});
                break;
            case internal_persistent:
                persistent_registrar.book(info.index_,
                        persistent_buffer_assigner_.query_size(info.index_));
                exec_args_set_.add_mem_use_internal_persistent(
                        {mem, info.index_});
                break;
            default: return status::unknown;
        }
    }

    // Prepare exec args for each op by using binded memories
    // TODO(qun) define each in/output's semantics in op def. Because the
    // semantics should be fixed and a part of IR
    ret = impl::topo_order_visit(
            impl::graph_t(subgraph).get_output_ops(), [&](impl::op_t *op) {
                if (op->get_kind() == op_kind::dnnl_convolution
                        || op->get_kind() == op_kind::dnnl_matmul
                        || op->get_kind() == op_kind::dnnl_convtranspose
                        || op->get_kind() == op_kind::dnnl_conv_depthwise) {
                    prepare_args_for_conv_and_matmul(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_pool
                        || op->get_kind() == op_kind::dnnl_softmax
                        || op->get_kind() == op_kind::dnnl_logsoftmax) {
                    const bool is_training = op->has_attr("is_training")
                            ? op->get_attr<bool>("is_training")
                            : false;
                    prepare_args_for_siso_op(
                            op, p_engine, mgr, true, is_training);
                } else if (op->get_kind() == op_kind::dnnl_softmax_bwd
                        || op->get_kind() == op_kind::dnnl_logsoftmax_bwd) {
                    prepare_args_for_softmax_bwd(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_pool) {
                    const bool is_training = op->has_attr("is_training")
                            ? op->get_attr<bool>("is_training")
                            : false;
                    prepare_args_for_dnnl_pool(
                            op, p_engine, mgr, true, is_training);
                } else if (op->get_kind() == op_kind::dnnl_pool_bwd) {
                    prepare_args_for_pool_bwd(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_concat) {
                    prepare_args_for_miso_op(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_eltwise
                        || op->get_kind() == op_kind::dnnl_reduction
                        || op->get_kind() == op_kind::dnnl_shuffle
                        || op->get_kind() == op_kind::permute
                        || op->get_kind() == op_kind::to_group
                        || op->get_kind() == op_kind::from_group
                        || op->get_kind() == op_kind::expand
                        || op->get_kind() == op_kind::squeeze
                        || op->get_kind() == op_kind::dnnl_resampling
                        || op->get_kind() == impl::op_kind::StaticReshape
                        || op->get_kind() == impl::op_kind::StaticTranspose) {
                    prepare_args_for_siso_op(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_prelu) {
                    prepare_args_for_prelu(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_prelu_bwd) {
                    prepare_args_for_prelu_bwd(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_bn_folding) {
                    bind_memory_for_bn_folding(op, p_engine);
                } else if (op->get_kind() == op_kind::dnnl_conv_bwd_data
                        || op->get_kind()
                                == op_kind::dnnl_convtranspose_bwd_data) {
                    bind_memory_for_conv_bwd_data(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_conv_bwd_weights
                        || op->get_kind()
                                == op_kind::dnnl_convtranspose_bwd_weights) {
                    bind_memory_for_conv_bwd_weights(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_batchnorm) {
                    bind_memory_for_batchnorm(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_batchnorm_bwd) {
                    bind_memory_for_batchnorm_bwd(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_layernorm) {
                    bind_memory_for_layernorm(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_layernorm_bwd) {
                    bind_memory_for_layernorm_bwd(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_sum) {
                    prepare_args_for_miso_op(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_binary) {
                    prepare_args_for_binary(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_constant_scales
                        || op->get_kind() == op_kind::dnnl_constant_zps) {
                    prepare_args_for_niso_op(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_mul_scales
                        || op->get_kind() == op_kind::dnnl_reorder) {
                    prepare_args_for_reorder_op(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_resampling_bwd) {
                    prepare_args_for_resampling_bwd(op, p_engine, mgr);
                } else if (op->get_kind() == op_kind::dnnl_eltwise_bwd) {
                    prepare_args_for_eltwise_bwd(op, p_engine, mgr);
                } else {
                    assertm(false, "memory planning: unsupported op");
                    return impl::status::compile_fail;
                }
                return impl::status::success;
            });
    if (ret != status::success) return ret;

    return status::success;
}

impl::status_t memory_planner_t::prepare_subgraph_inplace_pairs(
        std::shared_ptr<subgraph_t> &sg, bool enable_standard_sharing) {
    size_t time_point = 0;
    impl::topo_order_visit(sg->get_output_ops(), [&](op_t *cur_op) {
        auto out_vals = cur_op->get_output_values();
        for (auto &out_val : out_vals) {
            auto out_buf = buffer_assignments_.at(out_val.get());
            if (out_buf.kind_ != external_output) continue;
            impl::logical_tensor_t out_lt = sg->outs_[out_buf.index_];
            impl::logical_tensor_t in_lt = impl::zero_logical_tensor();

            // check if can inplaced sharing external input buffer
            bool inplace_shared = false;
            auto op_inplace_pairs
                    = get_op_inplace_pairs(*cur_op, sg->fusion_info_mgr_);
            for (const auto &pair : op_inplace_pairs) {
                if (pair.out_idx_ != out_val->get_offset()) continue;

                auto in_val = cur_op->get_input_value(pair.in_idx_);
                auto in_buf = buffer_assignments_.at(in_val.get());
                if (in_buf.kind_ != external_input) continue;

                in_lt = sg->ins_[in_buf.index_];
                inplace_shared = true;
                break;
            }

            // check if can standard sharing external input. note: from library
            // side, it's standard sharing, but from FWK side, it's inplace
            // sharing
            bool standard_shared = false;
            if (enable_standard_sharing && !inplace_shared) {
                std::vector<logical_tensor_t> candidates;
                for (auto &ex_in : external_inputs_live_range_) {
                    // external buffer is still in use
                    if (ex_in.second.end_ >= time_point) continue;

                    // different memory size, can't reuse
                    auto in_md = make_dnnl_memory_desc(
                            sg->ins_[ex_in.first->index_]);
                    auto out_md
                            = make_dnnl_memory_desc(sg->outs_[out_buf.index_]);
                    if (in_md.get_size() != out_md.get_size()) continue;

                    candidates.emplace_back(sg->ins_[ex_in.first->index_]);
                }

                // There may be multiple external input buffers that can be
                // shared with the external output buffer. we decided to only
                // report one pair now. To not break existing tests, we prefer
                // to choose the one whose logical tensor id is larger (the
                // post-src in test). We can change this criteria if we have any
                // real cases or requests in the future.
                if (!candidates.empty()) {
                    in_lt = candidates[0];
                    for (auto &tmp : candidates) {
                        if (tmp.id > in_lt.id) { in_lt = tmp; }
                    }
                    standard_shared = true;
                }
            }

            // No sharing
            if (!inplace_shared && !standard_shared) continue;

            // Have shared, not re-do
            bool have_shared = false;
            for (auto &pair : inplace_pairs_) {
                if (pair.output == out_lt.id || pair.input == in_lt.id)
                    have_shared = true;
            }
            if (have_shared) continue;

            // TODO(qun) we didn't report iplace pair if two lts have different
            // layout type because of frontend users didn't process this
            // situation at this moment. In the future, we need to fix this for
            // more inplace opportunities. Here the condition of alias_ins == 1
            // is to disable the inplace option for src = conv(src) + src
            ltw in_ltw(in_lt), out_ltw(out_lt);
            size_t alias_ins = 0;
            for (auto &tmp : sg->ins_) {
                if (in_ltw.id() == tmp.id) alias_ins++;
            }
            bool can_share = alias_ins == 1
                    && in_ltw.property_type() != impl::property_type::constant
                    && in_ltw.layout_type() == out_ltw.layout_type();
            if (can_share)
                inplace_pairs_.push_back({in_ltw.id(), out_ltw.id()});
        }
        time_point++;
        return impl::status::success;
    });

    return status::success;
}

// In this function, we will do the following things:
// - Build the alias map. both the key and value in the map are edges. the key
//   is the alias of value.
// - Count the reference count of each edges. the reference count will be used
//   during assign temporary buffer to determine which edge's buffer can be
//   reused since it ref count reduce to zero.
// - Assign external user given inputs/outputs buffer to corresponding edges
// - Assign internal allocated temporary buffer to corresponding edges.
// - Assign internal allocated persistent buffer to conresponding edges.
// - Prepare the memory objects which will be used in execution.
impl::status_t memory_planner_t::run(std::shared_ptr<subgraph_t> &sg) {
    status_t ret;

    auto &subgraph = sg->get_mutable_ops();
    auto &mgr = sg->fusion_info_mgr_;
    const auto &p_engine = *(sg->p_engine_);
    const auto &inputs = sg->ins_;
    const auto &outputs = sg->outs_;

    clear(); // clear state to make the method be reentrant

    alias_analyzer_.run(sg);

    // get the reference count of each edge
    std::unordered_map<value_t *, size_t> edge_ref_count;
    for (auto &cur_op : subgraph) {
        auto in_vals = cur_op->get_input_values();
        for (auto &val : in_vals) {
            edge_ref_count[val.get()]++;
        }
    }
    for (auto &val : impl::graph_t(subgraph).get_output_values()) {
        edge_ref_count[val]++;
    }

    // if not enable memory sharing, we add additional 1 to edge reference
    // count, so that tensors will not be reused
    if (!enable_memory_sharing_) {
        for (auto &val_count : edge_ref_count) {
            val_count.second++;
        }
    }

    // Assign external_input buffers to subgraph's inputs and their alias
    ret = assign_external_inputs_buffer(subgraph, inputs);
    if (ret != status::success) return ret;

    // Assign internal temporary buffer for all other edges
    ret = assign_internal_temporary_buffer(
            subgraph, edge_ref_count, mgr, false);
    if (ret != status::success) return ret;

    // Replace some internal temporary buffers to user given external output
    // buffer
    ret = assign_external_outputs_buffer(subgraph, outputs, mgr);
    if (ret != status::success) return ret;

    // Replace some internal temporary buffers to cached persistent buffer
    ret = assign_internal_persistent_buffer(subgraph, mgr);
    if (ret != status::success) return ret;

    // Reset the unreplaced internal temporary buffer
    temporary_buffer_assigner_.clear();
    for (auto it = buffer_assignments_.begin();
            it != buffer_assignments_.end();) {
        if (it->second.kind_ == internal_temporary) {
            it = buffer_assignments_.erase(it);
        } else {
            it++;
        }
    }

    // Re-assign internal temporary buffer for reset ones (will re-do memory
    // sharing between temporary buffers)
    ret = assign_internal_temporary_buffer(subgraph, edge_ref_count, mgr, true);
    if (ret != status::success) return ret;

    // Check which input/output pair of the subgraph can be inplaced
    ret = prepare_subgraph_inplace_pairs(sg, false);
    if (ret != status::success) return ret;

    // Bind memory object to each value
    ret = prepare_execution_args_set(subgraph, p_engine, mgr);
    if (ret != status::success) return ret;

    return impl::status::success;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
