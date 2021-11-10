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

#include <vector>
#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "oneapi/dnnl/dnnl_graph_sycl.hpp"

using namespace dnnl::graph;
using namespace cl::sycl;

struct api_engine_params {
    engine::kind eng_kind_;
};

/// Below functions aim to simulate what integration layer does
/// before creating dnnl::graph::allocator. We expect integration will
/// wrap the real SYCL malloc_device/free functions like below, then
/// dnnl_graph_allocator will be created with correct allocation/deallocation
/// function pointers.
void *sycl_malloc_wrapper(size_t n, const void *dev, const void *ctx,
        dnnl::graph::allocator::attribute attr) {
    namespace api = dnnl::graph;
    // need to handle different allocation types according to attr.type
    if (attr.type
            == api::allocator::convert_to_c(
                    api::allocator::lifetime::persistent)) {
        // persistent memory
    } else if (attr.type
            == api::allocator::convert_to_c(api::allocator::lifetime::output)) {
        // output tensor memory
    } else {
        // temporary memory
    }

    return malloc_device(n, *static_cast<const cl::sycl::device *>(dev),
            *static_cast<const cl::sycl::context *>(ctx));
}

void sycl_free_wrapper(void *ptr, const void *context) {
    free(ptr, *static_cast<const cl::sycl::context *>(context));
}

class api_engine : public ::testing::TestWithParam<api_engine_params> {
public:
    void create_with_sycl() {
        auto param = ::testing::TestWithParam<api_engine_params>::GetParam();
        engine::kind kind = param.eng_kind_;

        std::unique_ptr<device> sycl_dev;
        std::unique_ptr<context> sycl_ctx;

        auto platform_list = platform::get_platforms();
        for (const auto &plt : platform_list) {
            auto device_list = plt.get_devices();
            for (const auto &dev : device_list) {
                if ((kind == engine::kind::gpu && dev.is_gpu())
                        || (kind == engine::kind::cpu && dev.is_cpu())) {
                    sycl_dev.reset(new device(dev));
                    sycl_ctx.reset(new context(*sycl_dev));
                    engine e = dnnl::graph::sycl_interop::make_engine(
                            *sycl_dev, *sycl_ctx);
                    allocator alloc = dnnl::graph::sycl_interop::make_allocator(
                            sycl_malloc_wrapper, sycl_free_wrapper);
                    e.set_allocator(alloc);
                    ASSERT_EQ(e.get_device_id(), 0);
                }
            }
        }
    }
};

TEST_P(api_engine, create_with_sycl) {
    create_with_sycl();
}

#ifdef DNNL_GRAPH_GPU_SYCL
INSTANTIATE_TEST_SUITE_P(api_engine_gpu, api_engine,
        ::testing::Values(api_engine_params {engine::kind::gpu}));
#endif

#ifdef DNNL_GRAPH_CPU_SYCL
INSTANTIATE_TEST_SUITE_P(api_engine_cpu, api_engine,
        ::testing::Values(api_engine_params {engine::kind::cpu}));
#endif
