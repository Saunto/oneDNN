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

#include <iostream>

#include <gtest/gtest.h>

#include "unit_test_common.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
cl::sycl::device &get_device() {
    static cl::sycl::device dev
            = get_test_engine_kind() == impl::engine_kind::cpu
            ? cl::sycl::device {cl::sycl::cpu_selector {}}
            : cl::sycl::device {cl::sycl::gpu_selector {}};
    return dev;
}

cl::sycl::context &get_context() {
    static cl::sycl::context ctx {get_device()};
    return ctx;
}

void *sycl_alloc(size_t n, const void *dev, const void *ctx,
        impl::allocator_attr_t attr) {
    return cl::sycl::malloc_device(n,
            *static_cast<const cl::sycl::device *>(dev),
            *static_cast<const cl::sycl::context *>(ctx));
}
void sycl_free(void *ptr, const void *ctx) {
    return cl::sycl::free(ptr, *static_cast<const cl::sycl::context *>(ctx));
}
#endif // DNNL_GRAPH_WITH_SYCL

impl::engine_t &get_engine() {
    if (get_test_engine_kind() == impl::engine_kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        static auto sycl_allocator = std::shared_ptr<impl::allocator_t>(
                impl::allocator_t::create(sycl_alloc, sycl_free),
                [](impl::allocator_t *alloc) { alloc->release(); });
        static impl::engine_t eng(
                impl::engine_kind::cpu, get_device(), get_context());
        eng.set_allocator(sycl_allocator.get());
#else
        static impl::engine_t eng(impl::engine_kind::cpu, 0);
#endif
        return eng;
    } else {
#ifdef DNNL_GRAPH_GPU_SYCL
        static auto sycl_allocator = std::shared_ptr<impl::allocator_t>(
                impl::allocator_t::create(sycl_alloc, sycl_free),
                [](impl::allocator_t *alloc) { alloc->release(); });
        static impl::engine_t eng(
                impl::engine_kind::gpu, get_device(), get_context());
        eng.set_allocator(sycl_allocator.get());
#else
        assert(!"GPU only support DPCPP runtime now");
        static impl::engine_t eng(impl::engine_kind::gpu, 0);
#endif
        return eng;
    }
}

impl::stream_t &get_stream() {
    if (get_test_engine_kind() == impl::engine_kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        static cl::sycl::queue q {get_context(), get_device(),
                cl::sycl::property::queue::in_order {}};
        static impl::stream_t strm {&get_engine(), q};
#elif DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
        static impl::stream_t strm {
                &get_engine(), dnnl::graph::testing::get_threadpool()};
#else
        static impl::stream_t strm {&get_engine()};
#endif
        return strm;
    } else {
#ifdef DNNL_GRAPH_GPU_SYCL
        static cl::sycl::queue q {get_context(), get_device(),
                cl::sycl::property::queue::in_order {}};
        static impl::stream_t strm {&get_engine(), q};
#else
        assert(!"GPU only support DPCPP runtime now");
        static impl::stream_t strm {&get_engine()};
#endif
        return strm;
    }
}

static impl::engine_kind_t test_engine_kind;

impl::engine_kind_t get_test_engine_kind() {
    return test_engine_kind;
}

void set_test_engine_kind(impl::engine_kind_t kind) {
    test_engine_kind = kind;
}
