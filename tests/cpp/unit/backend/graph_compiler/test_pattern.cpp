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

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "backend/graph_compiler/compiler_backend.hpp"
#include "interface/graph.hpp"
#include "interface/partition.hpp"
#include "test_utils.hpp"
#include "utils/pm/pass_base.hpp"
#include "utils/pm/pass_manager.hpp"

namespace impl = dnnl::graph::impl;
namespace compiler_impl = dnnl::graph::impl::compiler_impl;
namespace pass = dnnl::graph::impl::pass;

pass::pass_base_ptr get_pass(compiler_impl::compiler_backend_t &backend_ptr,
        const std::string &pass_name) {
    auto pm = pass::pass_manager_t(backend_ptr.get_pass_registry());
    auto &passes = pm.get_passes();
    auto find = std::find_if(passes.begin(), passes.end(),
            [&pass_name](const pass::pass_base_ptr &p) -> bool {
                return p->get_pass_name() == pass_name;
            });
    if (find == passes.end()) { return nullptr; }
    return *find;
}

// test int8 MHA pattern (optimized graph)
TEST(GCPatternTests, MHAInt8Pattern) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, true, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 20);
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

// test fp32 MHA pattern
TEST(GCPatternTests, MHAFp32Pattern) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 14);
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

// test fp32 MHA pattern (no reshape)
TEST(GCPatternTests, MHAFp32PatternOptionalReshape) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    utils::construct_f32_MHA(&agraph);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(p.num_ops(), 13);
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

// test bf16 MHA pattern
TEST(GCPatternTests, MHABf16Pattern) {
    REQUIRE_AVX512();
    impl::graph_t agraph;
    add_MHA_subgraph(&agraph, false, true, true);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();

    // it shall not match fp32 pass
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "fp32_mha_pattern");

    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 0);

    apass = get_pass(compiler_backend_ptr, "bf16_mha_pattern");
    REQUIRE_BF16_AMXBF16();
    apass->run(agraph);
    partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);

    impl::partition_t p;
    p.init(partitions[0]);

    auto partition_inputs = p.get_inputs();
    auto partition_outputs = p.get_outputs();
    ASSERT_EQ(partition_inputs.size(), 5);
    ASSERT_EQ(partition_outputs.size(), 1);
}

// test MHA pattern matcher v2 on graph variations
TEST(GCPatternTests, MHAInt8PatternVariation1) {
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    get_int8_MHA_subgraph_varients(&agraph);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions[0]->get_ops().size(), 20);
}

TEST(GCPatternTests, MHAInt8PatternVariation2) {
    // replace divide with multiply
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    get_int8_MHA_subgraph_varients(&agraph, false);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions[0]->get_ops().size(), 20);
}

TEST(GCPatternTests, MHAInt8PatternVariation3) {
    // set rescale output as Add's second input
    REQUIRE_VNNI_AMXINT8();
    impl::graph_t agraph;
    get_int8_MHA_subgraph_varients(&agraph, true,
            std::vector<quantize_position_t>(4, RESHAPE_INCLUDED), 1);
    agraph.build_graph();

    auto &compiler_backend_ptr
            = compiler_impl::compiler_backend_t::get_singleton();
    pass::pass_base_ptr apass
            = get_pass(compiler_backend_ptr, "int8_mha_pattern");
    apass->run(agraph);
    auto partitions = agraph.get_partitions();
    ASSERT_EQ(partitions.size(), 1);
    ASSERT_EQ(partitions[0]->get_ops().size(), 20);
}
