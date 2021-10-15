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

#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.h"

TEST(c_api_test, constant_cache) {
    int flag = 0;
    ASSERT_EQ(dnnl_graph_get_constant_cache(&flag), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_set_constant_cache(1), dnnl_graph_result_success);
    ASSERT_EQ(dnnl_graph_get_constant_cache(&flag), dnnl_graph_result_success);
    ASSERT_EQ(flag, 1);

    // negative test
    ASSERT_EQ(dnnl_graph_get_constant_cache(nullptr),
            dnnl_graph_result_error_invalid_argument);
    ASSERT_EQ(dnnl_graph_set_constant_cache(-1),
            dnnl_graph_result_error_invalid_argument);
}
