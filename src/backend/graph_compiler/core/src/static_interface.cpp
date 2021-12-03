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

/**
 * This file is useful in static build to make sure the initializers are
 * referenced and prevent them from being removed by the linker.
 * */

#define RUN_ON_PRODUCTION(F) \
    F(batch_matmul); \
    F(softmax); \
    F(tensor_view); \
    F(reshape); \
    F(matmul); \
    F(quantize); \
    F(dynamic_transpose);

#define DECL_INIT(NAME) extern volatile bool __help_dummy_##NAME;

#define RUN_ON_REFLECTION_CLASS(F) F(shared_ptr_static_data)

namespace sc {
RUN_ON_PRODUCTION(DECL_INIT)

#define DECL_REFLECTION_INIT(NAME) extern void *__reflection_init_##NAME;
namespace reflection {
RUN_ON_REFLECTION_CLASS(DECL_REFLECTION_INIT)
}

void __dummy_init() {
#define REF_INIT(NAME) (void)__help_dummy_##NAME;
    RUN_ON_PRODUCTION(REF_INIT)

#define REF_REFLECTION_INIT(NAME) (void)reflection::__reflection_init_##NAME;
    RUN_ON_REFLECTION_CLASS(REF_REFLECTION_INIT)
}

} // namespace sc
