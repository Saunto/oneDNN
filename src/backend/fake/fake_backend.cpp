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

#include <utility>

#include "utils/compatible.hpp"

#include "fake_backend.hpp"
#include "single_op_pass.hpp"
#include "transformation_pass.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace fake_impl {

fake_backend_t::fake_backend_t(const std::string &name, float priority)
    : backend(name, priority) {
    bool ret = register_passes();
    if (!ret) { throw std::runtime_error(name + " initialize failed"); }
}

bool fake_backend_t::register_passes() {
    FAKE_BACKEND_REGISTER_PASSES_CALL(single_op_pass, pass_registry_);
    return true;
}

} // namespace fake_impl

// This function must be called by backend_registry_t
void register_fake_backend() {
    backend_registry_t::get_singleton().register_backend(
            &fake_impl::fake_backend_t::get_singleton());
}

} // namespace impl
} // namespace graph
} // namespace dnnl
