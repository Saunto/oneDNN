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
#include "backend/graph_compiler/compiler_backend.hpp"
#include "interface/graph.hpp"
#include "interface/partition.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "cpp/unit/unit_test_common.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

namespace impl = dnnl::graph::impl;

// test fp32 get partition + compile + execution of MHA graph
TEST(GCGraphTest, Fp32MHACompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);
    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs;
    std::vector<const impl::logical_tensor_t *> outputs;
    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }
    impl::compiled_partition_t cp(p);
    impl::engine_t &eng = get_engine();
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    std::vector<impl::tensor_t> execution_inputs;
    std::vector<impl::tensor_t> execution_outputs;
    size_t size = 0;
    for (auto &lt : partition_inputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    test::vector<char> data(size);
    ASSERT_EQ(size, 805502980); // assertion on total tensor buffer size

    size = 0;
    for (auto &lt : partition_inputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_inputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_outputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, execution_inputs, execution_outputs),
            impl::status::success);
    strm.wait();
}

// test int8 get partition + compile + execution of MHA graph
TEST(GCGraphTest, Int8MHACompileExecution) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs;
    std::vector<const impl::logical_tensor_t *> outputs;

    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    impl::engine_t &eng = get_engine();
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    std::vector<impl::tensor_t> execution_inputs;
    std::vector<impl::tensor_t> execution_outputs;

    size_t size = 0;
    for (auto &lt : partition_inputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    test::vector<char> data(size);
    ASSERT_EQ(size, 201523204); // assertion on total tensor buffer size

    size = 0;
    for (auto &lt : partition_inputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_inputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_outputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, execution_inputs, execution_outputs),
            impl::status::success);
    strm.wait();
}

// test bf16 get partition + compile + execution of MHA graph
TEST(GCGraphTest, Bf16MHACompileExecution) {
    REQUIRE_BF16_AMXBF16();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, false, true, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);

    std::vector<const impl::logical_tensor_t *> inputs;
    std::vector<const impl::logical_tensor_t *> outputs;

    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    impl::engine_t &eng = get_engine();
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    std::vector<impl::tensor_t> execution_inputs;
    std::vector<impl::tensor_t> execution_outputs;

    size_t size = 0;
    for (auto &lt : partition_inputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    test::vector<char> data(size);
    ASSERT_EQ(size, 805502980 / 2); // assertion on total tensor buffer size

    size = 0;
    for (auto &lt : partition_inputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_inputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_outputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, execution_inputs, execution_outputs),
            impl::status::success);
    strm.wait();
}

// test infer shape + compile + execute for fp32 MHA
// the created graph is without intermediate shapes
// if infer shape failed, the compilation will also fail
TEST(GCGraphTest, Fp32MHAInfershapeCompileExecution) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_infer_shape(&agraph);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);
    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    std::vector<const impl::logical_tensor_t *> inputs;
    std::vector<const impl::logical_tensor_t *> outputs;
    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    impl::engine_t &eng = get_engine();
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    std::vector<impl::tensor_t> execution_inputs;
    std::vector<impl::tensor_t> execution_outputs;
    size_t size = 0;
    for (auto &lt : partition_inputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    test::vector<char> data(size);
    ASSERT_EQ(size, 805502980);

    size = 0;
    for (auto &lt : partition_inputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_inputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_outputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, execution_inputs, execution_outputs),
            impl::status::success);
    strm.wait();
}

// test compile + multithreading execution for fp32 MHA
TEST(GCGraphTest, Fp32MHACompileExecutionMultiThreading) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);
    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    std::vector<const impl::logical_tensor_t *> inputs;
    std::vector<const impl::logical_tensor_t *> outputs;
    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }

    impl::compiled_partition_t cp(p);
    impl::engine_t &eng = get_engine();
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    size_t total_buffer_size = 0;
    for (auto &lt : partition_inputs) {
        total_buffer_size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        total_buffer_size += compiler_backend_ptr.get_mem_size(lt);
    }

    int thread_num = 8;

    auto thread_func = [&](size_t tid) {
        test::vector<char> data(total_buffer_size);

        std::vector<impl::tensor_t> execution_inputs;
        std::vector<impl::tensor_t> execution_outputs;

        size_t size = 0;
        for (auto &lt : partition_inputs) {
            impl::tensor_t placeholder(lt, &eng, data.data() + size);
            execution_inputs.push_back(placeholder);
            size += compiler_backend_ptr.get_mem_size(lt);
        }
        for (auto &lt : partition_outputs) {
            impl::tensor_t placeholder(lt, &eng, data.data() + size);
            execution_outputs.push_back(placeholder);
            size += compiler_backend_ptr.get_mem_size(lt);
        }
        impl::stream_t &strm = get_stream();
        ASSERT_EQ(cp.execute(&strm, execution_inputs, execution_outputs),
                impl::status::success);
    };

    std::vector<std::thread> workers;
    for (size_t t_num = 0; t_num < thread_num; t_num++) {
        workers.emplace_back(thread_func, t_num);
    }

    for (size_t t_num = 0; t_num < thread_num; t_num++) {
        workers[t_num].join();
    }
}

// test allocator release before compiled partition destruction
TEST(GCGraphTest, AllocatorEarlyRelease) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();
    compiler_backend_ptr.get_partitions(agraph, impl::partition_policy::fusion);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);
    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();

    std::vector<const impl::logical_tensor_t *> inputs;
    std::vector<const impl::logical_tensor_t *> outputs;
    for (auto &lt : partition_inputs) {
        inputs.push_back(&lt);
    }
    for (auto &lt : partition_outputs) {
        outputs.push_back(&lt);
    }
    impl::compiled_partition_t cp(p);
    impl::allocator_t *allocator = impl::allocator_t::create();
    impl::engine_t eng(impl::engine_kind::cpu,
            0); // create a new engine rather than use test engine here to
    // avoid release the default allocator of the test engine
    eng.set_allocator(allocator);
    ASSERT_EQ(p.compile(&cp, inputs, outputs, &eng), impl::status::success);

    allocator->release(); // release the allocator

    std::vector<impl::tensor_t> execution_inputs;
    std::vector<impl::tensor_t> execution_outputs;
    size_t size = 0;
    for (auto &lt : partition_inputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    test::vector<char> data(size);

    size = 0;
    for (auto &lt : partition_inputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_inputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }
    for (auto &lt : partition_outputs) {
        impl::tensor_t placeholder(lt, &eng, data.data() + size);
        execution_outputs.push_back(placeholder);
        size += compiler_backend_ptr.get_mem_size(lt);
    }

    impl::stream_t &strm = get_stream();
    ASSERT_EQ(cp.execute(&strm, execution_inputs, execution_outputs),
            impl::status::success);
    strm.wait();
}
