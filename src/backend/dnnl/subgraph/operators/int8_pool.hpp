/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef BACKEND_DNNL_SUBGRAPH_OPERATORS_INT8_POOL_HPP
#define BACKEND_DNNL_SUBGRAPH_OPERATORS_INT8_POOL_HPP

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"
#include "backend/dnnl/legacy.hpp"
#include "backend/dnnl/subgraph/compile_ops.hpp"
#include "backend/dnnl/subgraph/infer_type.hpp"
#include "backend/dnnl/subgraph/layout_propagation.hpp"
#include "backend/dnnl/subgraph/memory_binding.hpp"
#include "backend/dnnl/subgraph/op_executable.hpp"
#include "backend/dnnl/subgraph/passes.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

struct quantized_pooling : public kernel_base {
private:
    dnnl::engine p_engine_;
    dnnl::stream p_stream_;
    impl::allocator_t *g_alloc_;

    std::vector<impl::op_t *> subgraph_;

    primitive_attr_mgr prm_attr_mgr_;
    execution_args_mgr exec_arg_mgr_;
    executable_mgr exec_mgr_;

    std::vector<op_executable *> execs_;
    std::vector<exec_args> exec_args_;

    char *val2 = std::getenv("DNNL_GRAPH_HOLD_MEMORY");
    bool enable_hold_memory_ = (val2 != nullptr && std::strcmp(val2, "1") == 0);

    bool first_iter_ {true};

    std::vector<std::shared_ptr<impl::op_t>> opt_subgraph_;

    char *internal_buf_ {nullptr};

public:
    virtual ~quantized_pooling() {
        if (!enable_hold_memory_ || !internal_buf_) return;
        allocator::free(internal_buf_, p_engine_, g_alloc_);
    }

    virtual impl::status_t compile_impl(const dnnl_partition_impl_t *part,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        // TODO(wuxun): since oneDNN pooling primitive only support u8u8 or
        // s8s8 on CPU device for now, we need to check whether the data types
        // between input and output are compatible. If we enable this check in
        // op schema or primitive supports u8s8/s8u8, then this check can be
        // safely removed.
        if (inputs[0].data_type != outputs[0].data_type)
            return status::unsupported;

        p_engine_ = make_dnnl_engine(*g_engine);
        g_alloc_ = g_engine->get_allocator();

        std::vector<std::shared_ptr<op_t>> subgraph = part->get_ops();

        // for those primitive ops like pooling, it requires the scales and zps
        // between input tensor and output tensor are the same. So here, we
        // don't need to split Dequant and Quant ops firstly, it should be okay
        // to directly fuse Dequant and Quant into int8 pool op.
        fuse_to_int8_pool(subgraph);

        insert_permute(subgraph);
        insert_reorder(subgraph);

        // have to set the given inputs and outputs before infer shape and
        // compile
        set_given_inputs_outputs(subgraph, inputs, outputs);
        impl::graph_t agraph(subgraph);
        BACKEND_DNNL_CHECK(agraph.infer_shape());
        BACKEND_DNNL_CHECK(agraph.infer_shape());
        BACKEND_DNNL_CHECK(infer_type(agraph));
        BACKEND_DNNL_CHECK(
                layout_propagation(subgraph, p_engine_, prm_attr_mgr_));

        // fill layout information for outputs logical tensors
        for (size_t i = 0; i < outputs.size(); i++) {
            for (auto out_val : impl::graph_t(subgraph).get_output_values()) {
                auto compiled_lt = out_val->get_logical_tensor();
                if (compiled_lt.id == outputs[i].id) {
                    auto lt = const_cast<impl::logical_tensor_t *>(&outputs[i]);
                    auto md = make_dnnl_memory_desc(compiled_lt);
                    fill_layout_info(lt, md);
                }
            }
        }

        // bind the memory for each op
        BACKEND_DNNL_CHECK(memory_binding(subgraph, inputs, outputs, p_engine_,
                exec_arg_mgr_, prm_attr_mgr_));

        BACKEND_DNNL_CHECK(
                compile_ops(subgraph, p_engine_, prm_attr_mgr_, exec_mgr_));

        // topologically sort the prms and their args
        impl::topo_order_visit(impl::graph_t(subgraph).get_output_ops(),
                [this](impl::op_t *op) {
                    auto exec_key = op->get_attr<int64_t>("executable_key");
                    auto &exec = exec_mgr_.get_executable(exec_key);
                    execs_.emplace_back(exec.get());

                    auto arg_key = op->get_attr<int64_t>("execution_args_key");
                    auto &args = exec_arg_mgr_.get_args(arg_key);
                    exec_args_.emplace_back(args);
                    return impl::status::success;
                });

        opt_subgraph_ = subgraph;

        return impl::status::success;
    }

    virtual impl::status_t execute_impl(const dnnl_partition_impl_t *part,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(part);
        if (first_iter_) { p_stream_ = make_dnnl_stream(p_engine_, *g_stream); }

        // update the data of partition in/outputs args
        for (size_t i = 0; i < inputs.size(); i++) {
            exec_arg_mgr_.get_external_input_mem(i).set_data_handle(
                    inputs[i].get_data_handle());
        }
        for (size_t i = 0; i < outputs.size(); i++) {
            exec_arg_mgr_.get_external_output_mem(i).set_data_handle(
                    outputs[i].get_data_handle());
        }

        // allocate buffer for internal memory object
        if (!enable_hold_memory_ || first_iter_) {
            internal_buf_ = (char *)allocator::malloc(
                    exec_arg_mgr_.get_internal_mem_size(), p_engine_, g_alloc_);
            char *start = internal_buf_;
            for (auto &mem : exec_arg_mgr_.get_internal_mems()) {
                mem.set_data_handle((void *)start);
                start += mem.get_desc().get_size();
            }
        }

        for (size_t i = 0; i < execs_.size(); i++) {
            execs_[i]->execute(p_stream_, exec_args_[i]);
        }

        // free buffer for internal memory object
        if (!enable_hold_memory_) {
            allocator::free(internal_buf_, p_engine_, g_alloc_);
        }

        first_iter_ = false;
        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
