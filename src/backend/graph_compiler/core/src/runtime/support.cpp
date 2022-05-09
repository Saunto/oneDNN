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

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#ifdef SC_OMP_ENABLED
#include <omp.h>
#endif
#include "config.hpp"
#include <runtime/data_type.hpp>
#include <runtime/env_var.hpp>
#include <runtime/env_vars.hpp>
#include <runtime/logging.hpp>
#include <runtime/runtime.hpp>
#include <util/os.hpp>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

SC_MODULE(runtime.support)

using namespace sc;
extern "C" void print_float(float f) {
    printf("%f\n", f);
}

extern "C" void print_index(uint64_t f) {
    printf("%llu\n", static_cast<unsigned long long>(f)); // NOLINT
}

extern "C" void print_int(int f) {
    printf("%d\n", f);
}

extern "C" void print_str(char *f) {
    fputs(f, stdout);
}

extern "C" uint64_t boundary_check(
        const char *name, uint64_t idx, uint64_t acc_len, uint64_t tsr_len) {
    if (idx >= tsr_len || idx + acc_len > tsr_len) {
        fprintf(stderr,
                "Boundary check for tensor %s failed. idx=%llu acc_len=%llu "
                "tsr_len=%llu\n",
                name, static_cast<unsigned long long>(idx), // NOLINT
                static_cast<unsigned long long>(acc_len), // NOLINT
                static_cast<unsigned long long>(tsr_len)); // NOLINT
        abort();
    }
    return idx;
}

extern "C" void *sc_global_aligned_alloc(size_t sz, size_t align) {
    return aligned_alloc(align, (sz / align + 1) * align);
}

extern "C" void sc_global_aligned_free(void *ptr, size_t align) {
    aligned_free(ptr);
}

namespace sc {

namespace runtime {
size_t get_os_page_size() {
#ifdef _WIN32
    // fix-me: (win32) impl
    return 4096;
#else
    static size_t v = getpagesize();
    return v;
#endif
}
} // namespace runtime

runtime_config_t &runtime_config_t::get() {
    static runtime_config_t cfg {};
    return cfg;
}

#ifdef SC_OMP_ENABLED
int runtime_config_t::get_num_threads() {
    return omp_get_max_threads();
}

void runtime_config_t::set_num_threads(int num) {
    return omp_set_num_threads(num);
}
#else
int runtime_config_t::get_num_threads() {
    return 1;
}

void runtime_config_t::set_num_threads(int num) {}
#endif

using namespace env_key;
runtime_config_t::runtime_config_t() {
    trace_initial_cap_ = 2048 * 1024;
    trace_out_path_ = utils::getenv_string(env_names[SC_TRACE]);
    execution_verbose_
            = (utils::getenv_int(env_names[SC_EXECUTION_VERBOSE], 0) == 1);

    amx_exclusive_ = static_cast<bool>(
            utils::getenv_int(env_names[SC_AMX_EXCLUSIVE], 0));
    constexpr int default_verbose = 0;
    int tmp_get_verbose_level
            = utils::getenv_int(env_names[SC_VERBOSE], default_verbose);
    if (tmp_get_verbose_level < 0 || tmp_get_verbose_level > 2) {
        tmp_get_verbose_level = 0;
    }
    verbose_level_ = tmp_get_verbose_level;
}
} // namespace sc

extern "C" void sc_value_check(void *tsr, const char *name, size_t size) {
    // temporarily assume dtype is float32
    float *buf = reinterpret_cast<float *>(tsr);
    for (size_t i = 0; i < size / sizeof(float); i++) {
        float val = static_cast<float>(buf[i]);
        if (std::isnan(val) || std::isinf(val)) {
            SC_MODULE_WARN << "Invalid value (nan or inf) found in tensor "
                           << name << " idx=" << i;
        }
    }
}
