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

#ifndef BACKEND_DNNL_PATTERNS_TRANSFORMATION_PATTERN_HPP
#define BACKEND_DNNL_PATTERNS_TRANSFORMATION_PATTERN_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "backend/dnnl/dnnl_partition_impl.hpp"

#include "utils/pm/nested_matcher.hpp"
#include "utils/pm/pass_base.hpp"
#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pattern {

class pattern_utils_t {
public:
    inline void match(dnnl::graph::impl::graph_t &backend_graph,
            std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph,
            std::vector<std::vector<op_t *>> &fusion_ops);

    inline void init_partition(dnnl::graph::impl::graph_t &backend_graph,
            std::vector<std::vector<op_t *>> &fusion_ops,
            const FCreateKernel &kernel_creator,
            dnnl::graph::impl::partition_kind_t pkind,
            const std::unordered_set<size_t> &input_index_having_context = {});

    pattern_utils_t() = default;
    pattern_utils_t(const pattern_utils_t &) = delete;
    pattern_utils_t(pattern_utils_t &&) = delete;
    pattern_utils_t &operator=(const pattern_utils_t &) = delete;
};

inline void pattern_utils_t::match(dnnl::graph::impl::graph_t &backend_graph,
        std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph,
        std::vector<std::vector<op_t *>> &fusion_ops) {
    // dfs_visit graph, do pattern matching
    topo_order_visit(backend_graph.get_output_ops(), [&](op_t *cur_op) {
        std::vector<op_t *> candidate_fusion;
        if (!impl::utils::pm::match_pattern(cur_op, pgraph, candidate_fusion)) {
            return status::success;
        }
        fusion_ops.emplace_back(candidate_fusion);
        return status::success;
    });
}

inline void pattern_utils_t::init_partition(
        dnnl::graph::impl::graph_t &backend_graph,
        std::vector<std::vector<op_t *>> &fusion_ops,
        const FCreateKernel &kernel_creator,
        dnnl::graph::impl::partition_kind_t pkind,
        const std::unordered_set<size_t> &input_index_having_context) {
    for (auto &pairs : fusion_ops) {
        std::shared_ptr<dnnl_partition_impl_t> pimpl
                = std::make_shared<dnnl_partition_impl_t>(
                        backend_graph.get_engine_kind(),
                        backend_graph.get_fpmath_mode(), pkind);
        pimpl->set_input_index_having_context(input_index_having_context);

        // transfer the matched op's ownership from graph to partition
        for (size_t i = 0; i < pairs.size(); ++i) {
            pimpl->add_op(pairs[i]->shared_from_this());
            // claim the op belong to the partition
            pairs[i]->set_partition(pimpl.get());
        }
        pimpl->init(kernel_creator);
        backend_graph.add_partition(pimpl);
    }
}

/*!
 * \brief transformation_pass_t generates an optimized graph
 *        when the pass is hit, it can be op replacements,
 *        dead branch elimination, etc.
 */
class transformation_pass_t : public impl::pass::pass_base {
public:
    explicit transformation_pass_t(std::string pbackend, std::string pname)
        : impl::pass::pass_base(std::move(pbackend), std::move(pname)) {}

    static impl::pass::pass_base_ptr create(
            std::string pbackend, std::string pname) {
        return std::make_shared<transformation_pass_t>(
                std::move(pbackend), std::move(pname));
    }

    // the criteria of pass execution
    impl::status_t run(impl::graph_t &agraph) override {
        // check if current pattern pass can be run on current graph
        engine_kind_t graph_engine_kind = agraph.get_engine_kind();
        if (get_engine_kind() != engine_kind::any_engine
                && get_engine_kind() != graph_engine_kind)
            return impl::status::success;

        // we can have only one optimized pattern
        std::vector<impl::pass::FCreatePattern> pfuncs
                = get_attr<impl::pass::FCreatePattern>("FCreatePattern");

        FCreateKernel kernel_creator
                = get_attr<FCreateKernel>("FCreateKernel")[0];

        const auto &idx_set = get_input_index_having_context();

        pattern_utils_t pu;
        for (auto &pfunc : pfuncs) {
            std::shared_ptr<impl::utils::pm::pb_graph_t> pgraph
                    = std::make_shared<impl::utils::pm::pb_graph_t>("pgraph");
            pfunc(pgraph);

            // for each pattern. match it
            std::vector<std::vector<op_t *>> fusion_ops;
            pu.match(agraph, pgraph, fusion_ops);
            if (!fusion_ops.empty()) {
                // temporary solution here for showing which pattern matched
                if (impl::utils::getenv_int_user("DUMP", 0) > 0
                        || impl::utils::check_verbose_string_user(
                                "DUMP", "pattern")) {
                    printf("onednn_graph_verbose,info,pattern,hit,%s\n",
                            get_pass_name().c_str());
                    fflush(stdout);
                }

                pu.init_partition(agraph, fusion_ops, kernel_creator,
                        get_kind(), idx_set);
            }
        }
        return impl::status::success;
    }
};

#define DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN( \
        backend_name, pattern_name) \
    registry.register_pass( \
            #backend_name, #pattern_name, &transformation_pass_t::create)

#define DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(pattern_class_) \
    void register_##pattern_class_(impl::pass::pass_registry_t &registry) {
#define DNNL_BACKEND_REGISTER_PATTERN_DEF_END }

#define MAX_REPETITION 4
} // namespace pattern
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
