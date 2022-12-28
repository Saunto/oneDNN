#ifndef CPU_RV64V_JIT_PLATFORM_TRAITS_HPP
#define CPU_RV64V_JIT_PLATFORM_TRAITS_HPP

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

int get_platform_maxvl();
int get_platform_vcores();
int get_platform_vector_level_cache_size();
int get_platform_l1d_cache_size();
int get_platform_l1d_cache_line_size();
int get_platform_vregs_count();

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64V_JIT_PLATFORM_TRAITS_HPP