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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_INTERFACE_GENERALIZE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_INTERFACE_GENERALIZE_HPP

#include "../module_pass.hpp"

namespace sc {

/**
 * Generates a wrapper for each function, which accepts generic values as
 * parameters
 * */
class interface_generalizer_t : public module_pass_t {
public:
    const_ir_module_ptr operator()(const_ir_module_ptr m) override;
};

} // namespace sc

#endif
