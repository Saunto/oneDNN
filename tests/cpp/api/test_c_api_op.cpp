/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.h"

#include "test_api_common.h"

/**
 * TODO: cover more op kind
*/
TEST(c_api_test, create_op) {
    dnnl_graph_op_t *op = NULL;
    dnnl_graph_op_kind_t op_kind = kConvolution;

#define CREATE_OP_DESTROY \
    do { \
        dnnl_graph_op_destroy(op); \
        op = NULL; \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_op_create(&op, 1, op_kind, "conv2d"),
            dnnl_graph_result_success, CREATE_OP_DESTROY);
    CREATE_OP_DESTROY;
#undef CREATE_OP_DESTROY
}

TEST(c_api_test, op_attr) {
    dnnl_graph_op_t *op = NULL;
    dnnl_graph_op_t *matmul_op = NULL;
    dnnl_graph_op_kind_t op_kind = kConvolution;

#define OP_ATTR_DESTROY \
    do { \
        dnnl_graph_op_destroy(op); \
        op = NULL; \
        dnnl_graph_op_destroy(matmul_op); \
        matmul_op = NULL; \
    } while (0);

    ASSERT_EQ_SAFE(dnnl_graph_op_create(&op, 1, op_kind, "conv2d"),
            dnnl_graph_result_success, OP_ATTR_DESTROY);

    int64_t strides[] = {4, 4};
    const char *auto_pad = "same_upper";
    int64_t groups = 2;
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op, "strides",
                           dnnl_graph_attribute_kind_is, &strides, 2),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op, "auto_pad",
                           dnnl_graph_attribute_kind_s, auto_pad, 1),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(op, "groups",
                           dnnl_graph_attribute_kind_i, &groups, 1),
            dnnl_graph_result_success, OP_ATTR_DESTROY);

    ASSERT_EQ_SAFE(dnnl_graph_op_create(&matmul_op, 2, kMatMul, "matmul"),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    bool transpose_a = true;
    bool transpose_b = false;
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(matmul_op, "transpose_a",
                           dnnl_graph_attribute_kind_b, &transpose_a, 1),
            dnnl_graph_result_success, OP_ATTR_DESTROY);
    ASSERT_EQ_SAFE(dnnl_graph_op_add_attr(matmul_op, "transpose_b",
                           dnnl_graph_attribute_kind_b, &transpose_b, 1),
            dnnl_graph_result_success, OP_ATTR_DESTROY);

    OP_ATTR_DESTROY;
#undef OP_ATTR_DESTROY
}
