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

#include <compiler/jit/xbyak/sc_xbyak_jit_generator.hpp>

#include <iomanip>
#include <iostream>

namespace sc {
namespace sc_xbyak {

sc_xbyak_jit_generator::sc_xbyak_jit_generator()
    : Xbyak::CodeGenerator(Xbyak::DEFAULT_MAX_CODE_SIZE, Xbyak::AutoGrow) {}

void *sc_xbyak_jit_generator::get_func_address(
        const std::string &func_name) const {
    auto iter = func_name_to_address_.find(func_name);
    if (iter == func_name_to_address_.end()) {
        return nullptr;
    } else {
        return iter->second;
    }
}

} // namespace sc_xbyak
} // namespace sc
