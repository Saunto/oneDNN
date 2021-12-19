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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_CONTEXT_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_CONTEXT_HPP

#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <util/def.hpp>
namespace sc {
union generic_val;

namespace runtime {

struct engine;

struct engine_vtable_t {
    using alloc_t = void *(*)(engine *, size_t);
    using dealloc_t = void (*)(engine *, void *);
    alloc_t persistent_alloc;
    dealloc_t persistent_dealloc;
    alloc_t temp_alloc;
    dealloc_t temp_dealloc;
};

struct stream_vtable_t : public engine_vtable_t {
    using parallel_call_cpu_t
            = void (*)(void (*)(void *, void *, int64_t, generic_val *), void *,
                    void *, int64_t, int64_t, int64_t, generic_val *);
    parallel_call_cpu_t parallel_call;

    constexpr stream_vtable_t(alloc_t persist_alloc, dealloc_t persist_dealloc,
            alloc_t tmp_alloc, dealloc_t tmp_dealloc,
            parallel_call_cpu_t parallel_call_f)
        : engine_vtable_t {persist_alloc, persist_dealloc, tmp_alloc,
                tmp_dealloc}
        , parallel_call(parallel_call_f) {}
};

struct engine {
    engine_vtable_t *vtable_;
    engine(engine_vtable_t *vtable) : vtable_(vtable) {}
};

struct stream_t : public engine {
    stream_t(stream_vtable_t *vtable) : engine {vtable} {}
    stream_vtable_t *vtable() const {
        return static_cast<stream_vtable_t *>(vtable_);
    }
};

SC_API stream_t *get_default_stream();

} // namespace runtime

} // namespace sc

#endif
