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

#include "cfake_jit.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <compiler/codegen/codegen_c.hpp>
#include <compiler/config/target_machine.hpp>
#include <compiler/jit/jit.hpp>
#include <compiler/jit/symbol_resolver.hpp>
#include <runtime/config.hpp>
#include <runtime/memorypool.hpp> // to get the path of the runtime library
#include <unordered_map>
#include <util/string_utils.hpp>
#include <util/utils.hpp>

#ifdef _WIN32
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

SC_MODULE(cfakejit)
namespace sc {

static std::atomic<int32_t> cnt {0};

#ifdef _WIN32
std::shared_ptr<jit_module> cfake_jit::make_jit_module(
        const std::string &inpath, const std::string &outpath,
        statics_table_t &&globals, bool has_generic_wrapper) {
    // fix-me: (win32)
    throw std::runtime_error("make_jit_module().");
}

std::shared_ptr<jit_module> cfake_jit::make_jit_module(
        const_ir_module_ptr module, bool generate_wrapper) {
    // fix-me: (win32)
    throw std::runtime_error("make_jit_module().");
}

void *cfake_jit_module_t::get_address_of_symbol(const std::string &name) {
    // fix-me: (win32)
    throw std::runtime_error("get_address_of_symbol().");
}

cfake_jit_module_t::~cfake_jit_module_t() {
    // fix-me: (win32)
    throw std::runtime_error("~cfake_jit_module()");
}

#else

std::shared_ptr<jit_module> cfake_jit::make_jit_module(
        const std::string &inpath, const std::string &outpath,
        statics_table_t &&globals, bool has_generic_wrapper) {
    auto &home_path = utils::get_sc_home_path();
    if (home_path.empty()) {
        throw std::runtime_error("environment variable SC_HOME is not set");
    }
    if (utils::compiler_configs_t::get().print_gen_code_) {
        std::ifstream f(inpath);
        if (f.is_open()) std::cerr << f.rdbuf();
    }
    const std::string home_inc = home_path + "/src";
    // Mandatory compiler options...
    std::vector<std::string> option = {command_, "-I", home_inc, "-o", outpath,
            inpath, "-shared", "-fPIC", "-std=c++11", "-DSC_JIT_SOURCE=1"};
#if SC_PROFILING == 1
    option.emplace_back("-g");
#endif

    // Discretionary compiler options...
    const std::string &options_group
            = utils::compiler_configs_t::get().jit_cc_options_;

    std::vector<std::string> discretionary_options;
    if (options_group.empty() || (options_group == "default")) {
        discretionary_options = std::vector<std::string> {"-march=native"};
        assert(opt_level_ >= 0 && opt_level_ <= 3);
        discretionary_options.emplace_back("-O");
        discretionary_options.back() += std::to_string(opt_level_);
        const auto &envflags = utils::compiler_configs_t::get().cpu_jit_flags_;
        for (const auto &i : envflags) {
            discretionary_options.emplace_back(i);
        }

        if (debug_info_) { discretionary_options.emplace_back("-g"); }
    } else if (options_group == "xbyak-dev") {
        discretionary_options = std::vector<std::string> {"-O3",
                "-march=native",

                // Produce assembly that our (WIP) maint/dev/gnu-as-to-xbyak.py
                // script knows how to parse.
                "-x c++", "-save-temps=obj", "-fverbose-asm", "-masm=intel",

                // Suppress object code that's too complicated for us to
                // replicate via Xbyak (at least for the moment).
                // TODO(xxx): It's possible that some of the following options
                // are unnecessary for that goal.
                "-fno-unwind-tables", "-fno-asynchronous-unwind-tables",
                "-fno-rtti", "-fno-exceptions", "-fno-stack-protector"};
    } else {
        throw std::runtime_error(
                "Unsupported value for env var SC_JIT_CC_OPTIONS_GROUP.");
    }
    option.insert(option.end(), discretionary_options.begin(),
            discretionary_options.end());

    int exit_status;
    bool success
            = utils::create_process_and_await(command_, option, exit_status);
    void *compiled_module = nullptr;
    if (success) {
        if (exit_status) {
            std::ostringstream os;
            os << "c compiler returns non-zero code: " << exit_status;
            throw std::runtime_error(os.str());
        }
        compiled_module = dlopen(outpath.c_str(), RTLD_LAZY);
        if (!compiled_module) {
            std::ostringstream os;
            os << "dlopen: " << dlerror();
            throw std::runtime_error(os.str());
        }
        for (auto &kv : get_runtime_function_map()) {
            void **ptr = reinterpret_cast<void **>(
                    dlsym(compiled_module, (kv.first + "_fptr").c_str()));
            if (ptr) { *ptr = kv.second; }
        }
        typedef void (*init_func_t)(void *ctx, void *mod);
        auto init_func = reinterpret_cast<init_func_t>(
                dlsym(compiled_module, "__sc_init__"));
        if (init_func) { init_func(nullptr, globals.data_.data_); }
    } else {
        // If we call 'unlink', it will overwrite errno.
        const int fork_errno = errno;
        if (!utils::compiler_configs_t::get().keep_gen_code_) {
            unlink(inpath.c_str());
        }

        std::ostringstream os;
        os << "Error when fork: " << utils::get_error_msg(fork_errno);
        throw std::runtime_error(os.str());
    }
    return std::shared_ptr<cfake_jit_module_t>(
            new cfake_jit_module_t(compiled_module, inpath, outpath,
                    std::move(globals), has_generic_wrapper));
}

std::shared_ptr<jit_module> cfake_jit::make_jit_module(
        const_ir_module_ptr module, bool generate_wrapper) {
    std::stringstream name_maker;
    name_maker << getpid() << '_' << ++cnt;
    std::string unique_name = name_maker.str();

    // If we're invoking gcc/g++ with the "-save-temps=obj" option, we
    // want to have just one "." character in the name. Otherwise it
    // seems to get confused and the saved intermediate files lack the
    // unique_name part.
    const auto &tmpdir = utils::compiler_configs_t::get().temp_dir_;
    std::string outpath = tmpdir + "/cfake_jit_module-" + unique_name + ".so";
    std::string inpath = tmpdir + "/cfake_jit_module-" + unique_name + ".cpp";

    std::ofstream of(inpath);
    auto gen = create_c_generator(of, context_, generate_wrapper);
    auto new_mod = gen(module);
    of.close();
    auto &attr_table = *new_mod->attr_.get<std::shared_ptr<statics_table_t>>(
            ir_module_t::attr_key_t::MODULE_DATA_BUFFERS);

    return make_jit_module(
            inpath, outpath, std::move(attr_table), generate_wrapper);
}

void *cfake_jit_module_t::get_address_of_symbol(const std::string &name) {
    void *global_var = globals_.get_or_null(name);
    if (global_var) { return global_var; }
    return dlsym(module_, name.c_str());
}

cfake_jit_module_t::~cfake_jit_module_t() {
    if (module_) {
        dlclose(module_);

        if (!utils::compiler_configs_t::get().keep_gen_code_) {
            unlink(path_.c_str());
            unlink(src_path_.c_str());
        }

        module_ = nullptr;
    }
}

#endif

std::shared_ptr<jit_function_t> cfake_jit_module_t::get_function(
        const std::string &name) {
    std::string fname(name);
    void *fun = get_address_of_symbol(fname);
    void *wrapper = get_address_of_symbol(fname + "_0wrapper");
    if (fun || wrapper) {
        if (runtime_config_t::get().execution_verbose_) {
            return std::make_shared<cfake_jit_function_t>(
                    shared_from_this(), fun, wrapper, fname);
        } else {
            return std::make_shared<cfake_jit_function_t>(
                    shared_from_this(), fun, wrapper);
        }
    } else {
        return nullptr;
    }
}

void cfake_jit_function_t::call_generic(
        runtime::stream_t *stream, generic_val *args) const {
    assert(wrapper_ && "Trying to call 'call_generic' \
            on a jit funciton with no wrapper.");

    using functype = void (*)(runtime::stream_t *, void *, generic_val *);
    functype f = reinterpret_cast<functype>(wrapper_);
    if (runtime_config_t::get().execution_verbose_) {
        using namespace std::chrono;
        auto start = steady_clock::now();
        f(stream, module_->globals_.data_.data_, args);
        auto stop = steady_clock::now();
        double duration
                = static_cast<double>(
                          duration_cast<nanoseconds>(stop - start).count())
                / 1e6;
        std::cout << "Entry point: " << fname_ << ". Time elapsed: " << duration
                  << " ms" << std::endl;
    } else {
        f(stream, module_->globals_.data_.data_, args);
    }
}

void cfake_jit_function_t::call_generic(
        runtime::stream_t *stream, void *module_data, generic_val *args) const {
    assert(wrapper_ && "Trying to call 'call_generic' \
            on a jit funciton with no wrapper.");

    using functype = void (*)(runtime::stream_t *, void *, generic_val *);
    functype f = reinterpret_cast<functype>(wrapper_);
    f(stream, module_->globals_.data_.data_, args);
}

template <typename T, typename TF>
constexpr uintptr_t myoffsetof(TF T::*fld) {
    return reinterpret_cast<uintptr_t>(&(*(T *)nullptr.*fld));
}
#define foffset(F) myoffsetof(&cpu_flags_t::F)

static std::unordered_map<std::string, uintptr_t> get_compiler_flag_map() {
    std::unordered_map<std::string, uintptr_t> ret = {
            {"__MMX__", foffset(fMMX)},
            {"__x86_64__", foffset(fx64)},
            {"__ABM__", foffset(fABM)},
            {"__RDRND__", foffset(fRDRAND)},
            {"__BMI__", foffset(fBMI1)},
            {"__BMI2__", foffset(fBMI2)},
            {"__ADX__", foffset(fADX)},
            {"__PREFETCHWT1__", foffset(fPREFETCHWT1)},

            {"__SSE__", foffset(fSSE)},
            {"__SSE2__", foffset(fSSE2)},
            {"__SSE3__", foffset(fSSE3)},
            {"__SSSE3__", foffset(fSSSE3)},
            {"__SSE4_1__", foffset(fSSE41)},
            {"__SSE4_2__", foffset(fSSE42)},
            {"__SSE4A__", foffset(fSSE4a)},
            {"__AES__", foffset(fAES)},
            {"__SHA__", foffset(fSHA)},

            {"__AVX__", foffset(fAVX)},
            {"__XOP__", foffset(fXOP)},
            {"__FMA__", foffset(fFMA3)},
            {"__FMA4__", foffset(fFMA4)},
            {"__AVX2__", foffset(fAVX2)},

            {"__AVX512F__", foffset(fAVX512F)},
            {"__AVX512CD__", foffset(fAVX512CD)},
            {"__AVX512PF__", foffset(fAVX512PF)},
            {"__AVX512ER__", foffset(fAVX512ER)},
            {"__AVX512VL__", foffset(fAVX512VL)},
            {"__AVX512BW__", foffset(fAVX512BW)},
            {"__AVX512DQ__", foffset(fAVX512DQ)},
            {"__AVX512IFMA__", foffset(fAVX512IFMA)},
            {"__AVX512VBMI__", foffset(fAVX512VBMI)},
            {"__AVX512BF16__", foffset(fAVX512BF16)},
    };
    return ret;
}

static bool &get_flag_field(cpu_flags_t &flg, uintptr_t diff) {
    return *(bool *)((char *)&flg + diff);
}

static cpu_flags_t get_compiler_flags(
        const std::unordered_map<std::string, uintptr_t> &flagmap) {
    cpu_flags_t ret;
    std::vector<std::string> option
            = {"g++", "-march=native", "-dM", "-E", "-x", "c++", "-"};
    int exit_status;
    std::string rstdout, rstdin; // empty input
    bool success = utils::create_process_and_await(
            "g++", option, exit_status, &rstdin, &rstdout);
    if (success && !exit_status) {
        for (auto &v : utils::string_split(rstdout, " ")) {
            if (!v.empty() && v[0] == '_') {
                auto itr = flagmap.find(v);
                if (itr != flagmap.end()) {
                    get_flag_field(ret, itr->second) = true;
                }
            }
        }
        target_machine_t::set_simd_length_and_max_cpu_threads(ret);
        return ret;
    }
    SC_WARN << "Cannot run g++ to detect SIMD length!\n";
    return ret;
}

void cfake_jit::set_target_machine(target_machine_t &tm) {
    static auto flg_map = get_compiler_flag_map();
    auto impl = []() {
        auto compiler_flg = get_compiler_flags(flg_map);
        return compiler_flg;
    };

    static cpu_flags_t f = impl();
    for (auto &itr : flg_map) {
        if (get_flag_field(tm.cpu_flags_, itr.second)
                != get_flag_field(f, itr.second)) {
            SC_MODULE_WARN << "The flag " << itr.first
                           << " is enabled on your hardware but is "
                              "disabled by the default compiler.";
        }
    }
    if (tm.cpu_flags_.max_simd_bits > f.max_simd_bits) {
        SC_MODULE_WARN << "The hardware max SIMD length is larger than the "
                          "compiler SIMD length";
    }
    bool vnni_enabled = tm.cpu_flags_.fAVX512VNNI;
    bool amx_bf16_enabled = tm.cpu_flags_.fAVX512AMXBF16;
    bool amx_tile_enabled = tm.cpu_flags_.fAVX512AMXTILE;
    bool amx_int8_enabled = tm.cpu_flags_.fAVX512AMXINT8;
    tm.cpu_flags_ = f;
    tm.cpu_flags_.fAVX512VNNI = vnni_enabled;
    tm.cpu_flags_.fAVX512AMXBF16 = amx_bf16_enabled;
    tm.cpu_flags_.fAVX512AMXTILE = amx_tile_enabled;
    tm.cpu_flags_.fAVX512AMXINT8 = amx_int8_enabled;
}

} // namespace sc
