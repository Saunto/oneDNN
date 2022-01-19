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
 ******************************************************************************/

#include "matmul.hpp"
#include <utility>
#include "compiler/ir/graph/fusible_op.hpp"

namespace sc {
namespace ops {

matmul_op::matmul_op(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    COMPILE_ASSERT((ins.size() == 2 || ins.size() == 3),
            "matmul inputs size should be 2(a, b) or 3(a, b, bias).");
    COMPILE_ASSERT((ins[0]->details_.get_plain_dims().size()
                           == ins[1]->details_.get_plain_dims().size()),
            "matrix a and matrix b shape should be equal.");
    COMPILE_ASSERT((outs.size() == 1), "matmul outputs size should be 1");
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    for (auto &op : info_.outputs_) {
        op->producer_owner_ = this;
    }
    attrs_ = attrs;
    op_name_ = "matmul";
}

static void transed_matmul(const std::shared_ptr<sc_graph_t> &graph,
        any_map_t &attrs, bool is_batch, const graph_tensor_ptr &ins0,
        const graph_tensor_ptr &ins1, graph_tensor_ptr &trans0,
        graph_tensor_ptr &trans1) {
    if (attrs.get_or_else("transpose_a", false)) {
        auto original_dims = ins0->details_.get_plain_dims();
        sc_dims transed_plain_dims(original_dims.begin(), original_dims.end());
        std::swap(transed_plain_dims[transed_plain_dims.size() - 1],
                transed_plain_dims[transed_plain_dims.size() - 2]);
        std::vector<int> axes {(int)transed_plain_dims.size() - 1,
                (int)transed_plain_dims.size() - 2};
        auto out = graph_tensor::make(transed_plain_dims,
                ins0->details_.get_format(), ins0->details_.dtype_);
        trans0 = graph->make("transpose", {ins0}, {out}, {{"axes", axes}})
                         ->get_outputs()[0];
        attrs.set("transpose_a", false);
    } else {
        attrs.set("transpose_a", false);
    }

    // if transpose_b is true: need to permute
    if (attrs.get_or_else("transpose_b", false)) {
        auto original_dims = ins1->details_.get_plain_dims();
        sc_dims transed_plain_dims(original_dims.begin(), original_dims.end());
        std::swap(transed_plain_dims[transed_plain_dims.size() - 1],
                transed_plain_dims[transed_plain_dims.size() - 2]);
        std::vector<int> axes {(int)transed_plain_dims.size() - 1,
                (int)transed_plain_dims.size() - 2};
        auto out = graph_tensor::make(transed_plain_dims,
                ins1->details_.get_format(), ins1->details_.dtype_);
        trans1 = graph->make("transpose", {ins1}, {out}, {{"axes", axes}})
                         ->get_outputs()[0];
        attrs.set("transpose_b", false);
    } else {
        attrs.set("transpose_b", false);
    }
}

std::shared_ptr<sc_graph_t> matmul_op::get_graph() {
    auto graph = std::make_shared<sc_graph_t>();
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    auto ins = graph->make_input(inputs);
    sc_op_ptr matmul, graph_out;

    // analysis matmul is matmul_core_op_t which is tunable op by
    // inputs[0](the left matrix) and inputs[1](the right matrix).
    graph_tensor_ptr trans0 = ins->get_outputs()[0],
                     trans1 = ins->get_outputs()[1];
    transed_matmul(graph, attrs_, false, ins->get_outputs()[0],
            ins->get_outputs()[1], trans0, trans1);

    bool is_bf16 = false;
    if (inputs[0]->details_.dtype_ == datatypes::bf16
            || inputs[1]->details_.dtype_ == datatypes::bf16
            || outputs[0]->details_.dtype_ == datatypes::bf16) {
        COMPILE_ASSERT(inputs[0]->details_.dtype_ == datatypes::bf16
                        && inputs[1]->details_.dtype_ == datatypes::bf16
                        && outputs[0]->details_.dtype_ == datatypes::bf16,
                "All inputs should have same data type.")
        is_bf16 = true;
    }

    matmul = graph->make("matmul_core", {trans0, trans1}, {}, {});
    if (is_bf16) {
        matmul = graph->make("cast", matmul->get_outputs(), {},
                {{"dtype", datatypes::bf16}});
    }

    // check optional input lotgical tensor: bias
    if (info_.inputs_.size() == 3) {
        // create bias op by using broadcast op
        // considering: {bs0, bs1, .., M, N} and {M,N}, for bias, it shape is
        // equal with N.
        if (is_bf16) {
            COMPILE_ASSERT(inputs[2]->details_.dtype_ == datatypes::bf16,
                    "All inputs should have same data type.")
        }
        int last_axis = inputs[0]->details_.get_plain_dims().size() - 1;
        auto bias = graph->make("add",
                {matmul->get_outputs()[0], ins->get_outputs()[2]}, {},
                {{"bc_axis", std::vector<int> {last_axis}}});
        graph->make_output(bias->get_outputs());
    } else {
        graph->make_output(matmul->get_outputs());
    }
    return graph;
} // namespace ops

void matmul_op::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {}

} // namespace ops

// matmul op is graph op, matmul_core_op_t is tunable op
OP_REGISTER(::sc::ops::matmul_op, matmul)
} // namespace sc
