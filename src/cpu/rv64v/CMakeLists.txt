set(RV64VJIT_LIB ${DNNL_LIBRARY_NAME}_cpu_rv64v_jit)
set(RV64VJIT_LIB_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/rvjit/rvjit.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/rvjit/rvjit.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/rvjit/rvjit.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rvjit/mnemonic.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/rvjit/instruction_types.h"
)
add_library(${RV64VJIT_LIB} OBJECT ${RV64VJIT_LIB_SOURCES})

set(RV64V_LIB ${DNNL_LIBRARY_NAME}_cpu_rv64v)
set(RV64V_LIB_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/rv64_gemm_convolution.cpp"    
    "${CMAKE_CURRENT_SOURCE_DIR}/rv64_gemm_convolution.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rv64_gemm_convolution_naive.cpp"    
    "${CMAKE_CURRENT_SOURCE_DIR}/rv64_gemm_convolution_naive.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rv64_gemm_convolution_jit.cpp"    
    "${CMAKE_CURRENT_SOURCE_DIR}/rv64_gemm_convolution_jit.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit_rv64v_convolution.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit_rv64v_convolution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/platform_traits.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/platform_traits.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/jit_assembler.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/convolution/driver.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/convolution/driver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/convolution/kernel.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/convolution/kernel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/convolution/gemm_driver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/convolution/gemm_driver.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/convolution/gemm_kernel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/jit/convolution/gemm_kernel.hpp"

    )
add_library(${RV64V_LIB} OBJECT ${RV64V_LIB_SOURCES})
target_link_libraries(${RV64V_LIB} PUBLIC ${RV64VJIT_LIB} "$<TARGET_OBJECTS:${RV64VJIT_LIB}>")

set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS $<TARGET_OBJECTS:${RV64V_LIB}>)
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS $<TARGET_OBJECTS:${RV64VJIT_LIB}>)