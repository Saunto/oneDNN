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

#include <memory>

#include <gtest/gtest.h>

#include "interface/op.hpp"
#include "interface/partition.hpp"

#include "backend/dnnl/dnnl_partition_impl.hpp"
#include "backend/fake/fake_partition_impl.hpp"

using namespace dnnl::graph::impl;

TEST(partition_test, create_simple) {
    dnnl_impl::dnnl_partition_impl_t p(engine_kind::cpu);
    ASSERT_EQ(p.get_ops().size(), 0);
}

TEST(partition_test, add_ops) {
    dnnl_impl::dnnl_partition_impl_t p(engine_kind::cpu);
    size_t id = 100;
    std::shared_ptr<op_t> n(new op_t(id, op_kind::Wildcard, "Wildcard"));
    p.add_op(n);
    ASSERT_EQ(p.get_ops().size(), 1);

    std::vector<size_t> ids {101, 102};
    std::vector<std::shared_ptr<op_t>> ops;
    for (auto id : ids) {
        ops.emplace_back(new op_t(id, op_kind::Wildcard, "Wildcard"));
    }

    p.add_op(ops);
    ASSERT_EQ(p.get_ops().size(), 3);
}

TEST(partition_test, get_ops) {
    dnnl_impl::dnnl_partition_impl_t p(engine_kind::cpu);
    size_t id = 100;
    std::shared_ptr<op_t> n(new op_t(id, op_kind::Wildcard, "Wildcard"));
    p.add_op(n);
    auto ops = p.get_ops();
    ASSERT_EQ(ops.size(), 1);
    ASSERT_EQ(ops[0]->get_id(), 100);
}

TEST(partition_test, init) {
    // (todo)xinyu: improve engine test
    engine_t eng {};
    dnnl_impl::dnnl_partition_impl_t p(eng.kind());
    op_t n(op_kind::Convolution);
    n.set_attr<int64_t>("groups", 0);
    p.init(&n);
    ASSERT_TRUE(p.is_initialized());
    ASSERT_TRUE(p.get_assigned_backend()->get_name() != "fake_backend");
    ASSERT_EQ(p.get_fused_op()->get_kind(), op_kind::Convolution);
    ASSERT_TRUE(p.get_fused_op()->has_attr("groups"));
    ASSERT_EQ(p.get_fused_op()->get_attr<int64_t>("groups"), 0);
}

TEST(partition_test, copy) {
    engine_t eng {};
    dnnl_impl::dnnl_partition_impl_t p(eng.kind());
    op_t n(op_kind::Convolution);
    n.set_attr<int64_t>("groups", 0);
    p.init(&n);
    ASSERT_TRUE(p.is_initialized());
    ASSERT_TRUE(p.get_assigned_backend()->get_name() != "fake_backend");
    ASSERT_EQ(p.get_fused_op()->get_kind(), op_kind::Convolution);
    ASSERT_TRUE(p.get_fused_op()->has_attr("groups"));
    ASSERT_EQ(p.get_fused_op()->get_attr<int64_t>("groups"), 0);

    // copy the partition
    dnnl_impl::dnnl_partition_impl_t p_copy(p);
    op_t *p_op = const_cast<op_t *>(p_copy.get_fused_op().get());
    p_op->set_attr<int64_t>("groups", 1);
    ASSERT_EQ(p_copy.get_fused_op()->get_attr<int64_t>("groups"), 1);
    ASSERT_NE(p_copy.get_fused_op()->get_attr<int64_t>("groups"),
            p.get_fused_op()->get_attr<int64_t>("groups"));
}

TEST(partition_test, fake_partition) {
    fake_impl::fake_partition_impl_t p(engine_kind::cpu);
    size_t id = 100;
    std::shared_ptr<op_t> n(new op_t(id, op_kind::Wildcard, "Wildcard"));
    p.init(n.get());
    ASSERT_TRUE(p.is_initialized());
    ASSERT_TRUE(p.get_assigned_backend()->get_name() == "fake_backend");
    ASSERT_EQ(p.get_fused_op()->get_kind(), op_kind::Wildcard);

    // copy the partition
    fake_impl::fake_partition_impl_t p_copy(p);
    ASSERT_TRUE(p_copy.is_initialized());
    ASSERT_TRUE(p_copy.get_assigned_backend()->get_name() == "fake_backend");
    ASSERT_EQ(p_copy.get_fused_op()->get_kind(), op_kind::Wildcard);

    // clone
    std::shared_ptr<partition_impl_t> p_share = p.clone();
    fake_impl::fake_partition_impl_t *p_share_raw
            = dynamic_cast<fake_impl::fake_partition_impl_t *>(p_share.get());
    ASSERT_TRUE(p_share_raw->is_initialized());
    ASSERT_TRUE(
            p_share_raw->get_assigned_backend()->get_name() == "fake_backend");
    ASSERT_EQ(p_share_raw->get_fused_op()->get_kind(), op_kind::Wildcard);
}
