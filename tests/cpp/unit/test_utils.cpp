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
#include <string>
#include <gtest/gtest.h>

#include "utils/compatible.hpp"

#ifndef DNNL_GRAPH_SUPPORT_CXX17
TEST(utils_test, optional) {
    //init, copy constructor
    dnnl::graph::impl::utils::optional_impl<int64_t> o1, o2 = 2, o3 = o2;
    ASSERT_EQ(*o2, 2);
    ASSERT_EQ(*o3, 2);
    dnnl::graph::impl::utils::optional_impl<float> o4(
            dnnl::graph::impl::utils::nullopt),
            o5 = dnnl::graph::impl::utils::nullopt;
    ASSERT_EQ(o4.has_value(), false);
    ASSERT_EQ(o4, o5);
    EXPECT_THROW(o4.value(), std::logic_error);
    EXPECT_THROW(o1.value(), std::logic_error);
}
#endif

TEST(utils_test, any) {
    dnnl::graph::impl::utils::any a = 1;
    ASSERT_EQ(dnnl::graph::impl::utils::any_cast<int>(a), 1);
    int *i = dnnl::graph::impl::utils::any_cast<int>(&a);
    ASSERT_NE(i, nullptr);
    ASSERT_EQ(*i, 1);
    a = 3.14;
    ASSERT_EQ(dnnl::graph::impl::utils::any_cast<double>(a), 3.14);
    a = true;
    ASSERT_EQ(dnnl::graph::impl::utils::any_cast<bool>(a), true);
    dnnl::graph::impl::utils::any b;
    ASSERT_EQ(b.empty(), true);
    EXPECT_THROW(dnnl::graph::impl::utils::any_cast<int>(b),
            dnnl::graph::impl::utils::bad_any_cast);
}
