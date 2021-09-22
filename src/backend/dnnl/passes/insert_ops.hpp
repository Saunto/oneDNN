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
#ifndef BACKEND_DNNL_PASSES_INSERT_OPS_HPP
#define BACKEND_DNNL_PASSES_INSERT_OPS_HPP

#include <memory>
#include <vector>

#include "interface/c_types_map.hpp"

#include "lower_down.hpp"
#include "utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

void insert_reorder(std::vector<std::shared_ptr<op_t>> &subgraph);

void insert_permute(std::vector<std::shared_ptr<op_t>> &subgraph);

void insert_to_group_for_conv(std::vector<std::shared_ptr<op_t>> &subgraph);

/// Insert a transpose op for matmul's input tensors
///
/// Only valid for below scenarios:
/// (1) src or weight's ndims is greater than 1
/// (2) either `transpose_a` or `transpose_b` is true
void insert_transpose_for_matmul(std::vector<std::shared_ptr<op_t>> &subgraph);

/// Insert an expand-squeeze pair for matmul
///
/// The usage of expand op:
///     There maybe three scenarios as below:
///     (1) one of inputs (src or weight) has only one dimension, DNNL require
//      at two dimensions, so need to insert dim 1 before/after the current dim
///     (2) The batch dimensions of src and weight are not matched, need to
///     expand
///     (3) bias dimensions are not matched with dst, need to expand
///
/// The usage of squeeze op:
///     Only will be inserted when previously expand op(s) inserted
///     For example, considering two inputs [3,4]x[4], the second input will be
///     expanded into [4,1], hence the output shape should be [3,1]. However,
///     this is inconsistent with the results derived from the shape inference.
///     So we use squeeze here to remove the extra 1 dimension to produce output
///     with [3].
void insert_expand_and_squeeze_for_matmul(
        std::vector<std::shared_ptr<op_t>> &subgraph);

/// Insert an dnnl_u8_to_s8 op for matmul's weight tensor
///
/// Only valid for below scenarios:
/// src and weight's dtype are both uint8
void insert_u8_to_s8_for_matmul(std::vector<std::shared_ptr<op_t>> &subgraph,
        primitive_attr_mgr &prm_attr_mgr);

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
