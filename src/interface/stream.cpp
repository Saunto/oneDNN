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

#include <vector>

#include "oneapi/dnnl/dnnl_graph.h"
#include "oneapi/dnnl/dnnl_graph_sycl.h"
#include "oneapi/dnnl/dnnl_graph_threadpool.h"

#include "interface/c_types_map.hpp"
#include "interface/partition.hpp"
#include "interface/stream.hpp"

#include "utils/utils.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

using namespace dnnl::graph::impl;

///
/// \brief dnnl_graph_thread_pool_t
///
status_t DNNL_GRAPH_API dnnl_graph_thread_pool_create(
        thread_pool_t **created_thread_pool, int32_t num_threads) {
    *created_thread_pool = new thread_pool_t {num_threads};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_thread_pool_destroy(
        thread_pool_t *thread_pool) {
    delete thread_pool;
    return status::success;
}

///
/// \brief dnnl_graph_stream_attr_t
///
status_t DNNL_GRAPH_API dnnl_graph_stream_attr_create(
        stream_attr_t **created_stream_attr, thread_pool_t *thread_pool) {
    *created_stream_attr = new stream_attr_t {thread_pool};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_stream_attr_destroy(
        stream_attr_t *stream_attr) {
    delete stream_attr;
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_stream_create(
        stream_t **created_stream, const engine_t *engine) {
    if (engine->kind() == engine_kind::gpu) { return status::invalid_argument; }
#if DNNL_GRAPH_CPU_SYCL
    UNUSED(created_stream);
    UNUSED(engine);
    return status::invalid_argument;
#else
    *created_stream = new stream_t {engine};
    return status::success;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_sycl_interop_stream_create(
        stream_t **created_stream, const engine_t *engine, const void *queue) {
#if DNNL_GRAPH_WITH_SYCL
    if (utils::any_null(created_stream, engine, queue)) {
        return status::invalid_argument;
    }
    auto &sycl_queue = *static_cast<const cl::sycl::queue *>(queue);

    bool is_gpu_engine = engine->kind() == engine_kind::gpu;
    bool is_gpu_queue = sycl_queue.get_device().is_gpu();
    if (is_gpu_engine != is_gpu_queue) { return status::invalid_argument; }
    if (is_gpu_engine) {
#ifndef DNNL_GRAPH_GPU_SYCL
        return status::invalid_argument;
#endif
    } else {
#ifndef DNNL_GRAPH_CPU_SYCL
        return status::invalid_argument;
#endif
    }

    *created_stream = new stream_t {engine, sycl_queue};
    return status::success;
#else
    UNUSED(created_stream);
    UNUSED(engine);
    UNUSED(queue);
    return status::unsupported;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_threadpool_interop_stream_create(
        stream_t **created_stream, const engine_t *engine, void *threadpool) {
#if DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
    if (utils::any_null(created_stream, engine, threadpool)) {
        return status::invalid_argument;
    }
    auto tp = static_cast<dnnl::graph::threadpool_interop::threadpool_iface *>(
            threadpool);
    *created_stream = new stream_t {engine, tp};
    return status::success;
#else
    UNUSED(created_stream);
    UNUSED(engine);
    UNUSED(threadpool);
    return status::unsupported;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_threadpool_interop_stream_get_threadpool(
        stream_t *astream, void **threadpool) {
#if DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
    if (utils::any_null(astream, threadpool)) return status::invalid_argument;
    dnnl::graph::threadpool_interop::threadpool_iface *tp;
    auto status = astream->get_threadpool(&tp);
    if (status == status::success) *threadpool = static_cast<void *>(tp);
    return status;
#else
    UNUSED(astream);
    UNUSED(threadpool);
    return status::unsupported;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_stream_create_with_attr(
        stream_t **created_stream, const engine_t *engine,
        const stream_attr_t *attr) {
    *created_stream = new stream_t {engine, attr};
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_stream_create_sycl_with_attr(
        stream_t **created_stream, const engine_t *engine, const void *queue,
        const stream_attr_t *attr) {
#if DNNL_GRAPH_WITH_SYCL
    if (utils::any_null(created_stream, engine, queue, attr)) {
        return status::invalid_argument;
    }
    auto &sycl_queue = *static_cast<const cl::sycl::queue *>(queue);
    *created_stream = new stream_t {engine, sycl_queue, attr};
    return status::success;
#else
    UNUSED(created_stream);
    UNUSED(engine);
    UNUSED(queue);
    UNUSED(attr);
    return status::unsupported;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_stream_wait(stream_t *stream) {
    return stream->wait();
}

status_t DNNL_GRAPH_API dnnl_graph_stream_destroy(stream_t *stream) {
    delete stream;
    return status::success;
}
