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

#include "interface/allocator.hpp"
#include "interface/c_types_map.hpp"

#include <CL/sycl.hpp>

namespace impl = dnnl::graph::impl;

struct test_allocator_params {
    impl::engine_kind_t eng_kind_;
};

/// Below functions aim to simulate what integration layer does
/// before creating dnnl::graph::allocator. We expect integration will
/// wrap the real SYCL malloc_device/free functions like below, then
/// dnnl_graph_allocator will be created with correct allocation/deallocation
/// function pointers.
void *sycl_malloc_wrapper(size_t n, const void *dev, const void *ctx,
        dnnl::graph::impl::allocator_attr_t attr) {
    // need to handle different allocation types according to attr.type
    if (attr.type == dnnl::graph::impl::allocator_lifetime::persistent) {
        // persistent memory
    } else if (attr.type == dnnl::graph::impl::allocator_lifetime::output) {
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

class TestAllocator : public ::testing::TestWithParam<test_allocator_params> {
public:
    void default_sycl_allocator() {
        auto param
                = ::testing::TestWithParam<test_allocator_params>::GetParam();
        impl::engine_kind_t kind = param.eng_kind_;

        namespace sycl = cl::sycl;
        namespace impl = dnnl::graph::impl;
        impl::allocator_t &alloc = *impl::allocator_t::create();
        sycl::queue q = kind == impl::engine_kind::gpu
                ? sycl::queue {sycl::gpu_selector {},
                        sycl::property::queue::in_order {}}
                : sycl::queue {sycl::cpu_selector {},
                        sycl::property::queue::in_order {}};

        impl::allocator_t::attribute_t attr {
                impl::allocator_lifetime::persistent, 4096};
        void *mem_ptr = alloc.allocate(
                static_cast<size_t>(16), q.get_device(), q.get_context(), attr);
        ASSERT_NE(mem_ptr, nullptr);
        alloc.deallocate(mem_ptr, q.get_context());
        alloc.release();
    }

    void sycl_allocator() {
        namespace sycl = cl::sycl;

        auto param
                = ::testing::TestWithParam<test_allocator_params>::GetParam();
        impl::engine_kind_t kind = param.eng_kind_;

        std::unique_ptr<sycl::device> sycl_dev;
        std::unique_ptr<sycl::context> sycl_ctx;

        auto platform_list = sycl::platform::get_platforms();
        dnnl::graph::impl::allocator_t::attribute_t alloc_attr {
                dnnl::graph::impl::allocator_lifetime::persistent, 1024};
        ASSERT_EQ(alloc_attr.data.type,
                dnnl::graph::impl::allocator_lifetime::persistent);
        ASSERT_EQ(alloc_attr.data.alignment, 1024);
        dnnl::graph::impl::allocator_t &sycl_alloc = *impl::allocator_t::create(
                sycl_malloc_wrapper, sycl_free_wrapper);
        for (const auto &plt : platform_list) {
            auto device_list = plt.get_devices();
            for (const auto &dev : device_list) {
                if ((kind == impl::engine_kind::gpu && dev.is_gpu())
                        || (kind == impl::engine_kind::cpu && dev.is_cpu())) {
                    sycl_dev.reset(new sycl::device(dev));
                    sycl_ctx.reset(new sycl::context(*sycl_dev));

                    auto *mem_ptr = sycl_alloc.allocate(static_cast<size_t>(16),
                            *sycl_dev, *sycl_ctx, alloc_attr);
                    ASSERT_NE(mem_ptr, nullptr);
                    sycl_alloc.deallocate(mem_ptr, *sycl_ctx);
                }
            }
        }
        sycl_alloc.release();
    }
};

TEST_P(TestAllocator, DefaultSyclAllocator) {
    default_sycl_allocator();
}

TEST_P(TestAllocator, SyclAllocator) {
    sycl_allocator();
}

#ifdef DNNL_GRAPH_GPU_SYCL
INSTANTIATE_TEST_SUITE_P(SyclAllocatorGpu, TestAllocator,
        ::testing::Values(test_allocator_params {impl::engine_kind::gpu}));
#endif

#ifdef DNNL_GRAPH_CPU_SYCL
INSTANTIATE_TEST_SUITE_P(SyclAllocatorCpu, TestAllocator,
        ::testing::Values(test_allocator_params {impl::engine_kind::cpu}));
#endif
