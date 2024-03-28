#include "cpu/rv64v/jit/platform_traits.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

int get_platform_maxvl() {
    return getenv_int("DNNL_RV64_MAXVL", 1024*32);
}

int get_platform_vcores() {
    return getenv_int("DNNL_RV64_VCORES", 4);
}

int get_platform_vector_level_cache_size() {
    return getenv_int("DNNL_RV64_VLC", 1*1024*1024);
}

int get_platform_l1d_cache_size() {
    return getenv_int("DNNL_RV64_L1D", 32*1024);
}

int get_platform_l1d_cache_line_size() {
    return getenv_int("DNNL_RV64_L1D_LINE", 128);
}

int get_platform_vregs_count() {
    return 32;
}

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl