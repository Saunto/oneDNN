#include "cpu/rv64v/rv64_gemm_convolution.hpp"
#include "common/utils.hpp"
#include "cpu/gemm_convolution_utils.hpp"
    
#include <cstdlib>
#include <iostream>
#include <vector>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// im2col and col2im comes from Berkeley Vision's Caffe
// https://github.com/BVLC/caffe/blob/master/LICENSE

// Function to check if 'a' is greater than or equal to zero and less than 'b'
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename T>
void im2col_cpu(const T* data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                T* data_col) {  
  std::cout << "setting output and channel sizes" << std::endl;
  const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  const int channel_size = height * width;
  std::cout << "output_h: " << output_h << std::endl;
  std::cout << "output_w: " << output_w << std::endl;
  std::cout << "channel_size: " << channel_size << std::endl;
  for (int channel = channels; channel--; data_im += channel_size) {
    std::cout << "channel: " << channel << std::endl;
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      std::cout << "kernel_row: " << kernel_row << std::endl;
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        std::cout << "kernel_col: " << kernel_col << std::endl;
        int input_row = -pad_h + kernel_row;
        std::cout << "input_row: " << input_row << std::endl;
        for (int output_row = 0; output_row < output_h; output_row++) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_col = 0; output_col < output_w; output_col++) {
              std::cout << "output_col: " << output_col << std::endl;
              //int index = ((channel * kernel_h + kernel_row) * kernel_w + kernel_col) * (output_h * output_w) + (output_row * output_w) + output_col;
              *(data_col++) = 0; 
              std::cout << "set data_col to 0" << std::endl;
            }
          } else {
            std::cout << "else" << std::endl;
            int input_col = -pad_w + kernel_col;
            std::cout << "input_col: " << input_col << std::endl;
            for (int output_col = 0; output_col < output_w; output_col++) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename T>
void col2im_cpu(const T* data_col, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                T* data_im) {
  std::fill_n(data_im, height * width * channels, T(0)); // Initializes or resets data_im to zeros
  const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row;
        for (int output_row = 0; output_row < output_h; output_row++) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w; // Skip the entire row in the column buffer if out of bounds
          } else {
            int input_col = -pad_w + kernel_col;
            for (int output_col = 0; output_col < output_w; output_col++) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}
/*
// Explicit instantiation to avoid linker errors for undefined references
template void im2col_cpu<float>(const float* data_im, const int channels,
                                const int height, const int width, const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
                                 const int height, const int width, const int kernel_h, const int kernel_w,
                                 const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                 double* data_col);

template void col2im_cpu<float>(const float* data_col, const int channels,
                                const int height, const int width, const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
                                 const int height, const int width, const int kernel_h, const int kernel_w,
                                 const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                 double* data_im);
*/
// GEMM: A * B = C
// https://github.com/soniab/darknetRISCVV/

void gemm_nn(int M, int N, int K, float ALPHA, 
        const float *A, int lda, 
        const float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        const float *A, int lda, 
        const float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        const float *A, int lda, 
        const float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        const float *A, int lda, 
        const float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        const float *A, int lda, 
        const float *B, int ldb,
        float BETA,
        float *C, int ldc){
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        const float *A, int lda, 
        const float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

using namespace dnnl::impl::memory_tracking::names;

template <data_type_t T>
status_t
rv64_gemm_convolution_fwd_t<T>::do_execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t* const, DNNL_ARG_SRC);
    auto wei = CTX_IN_MEM(const data_t* const, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t*, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t* const, DNNL_ARG_DST);
    auto const src_mb_size = pd()->IC() * pd()->IH() * pd()->IW();
    auto const dst_mb_size = pd()->OC() * pd()->OH() * pd()->OW();
    std::cout << "rv64_gemm_convolution_fwd_t<T>::do_execute" << std::endl;
    
    // Use a scratchpad to hold the intermediate data
    auto scratchpad = ctx.get_scratchpad_grantor();
    float col[100];//scratchpad.get<float>(key_conv_gemm_col);
    std::cout << "Before im2col" << std::endl;
    // im2col
    im2col_cpu(src, pd()->IC(), pd()->IH(), pd()->IW(), 
        pd()->KH(), pd()->KW(), 1, 1, 
        1 , 1, col);
    std::cout << "After im2col" << std::endl;
    // GEMM: A * B = C
    // A is the input tensor, B is the weight tensor, and C is the output tensor

     // Calculate dimensions for GEMM based on convolution parameters
    int M = pd()->OC(); // Number of output channels
    int N = pd()->OH() * pd()->OW(); // Output spatial dimensions (flattened)
    int K = pd()->IC() * pd()->KH() * pd()->KW(); // Dimension shared by A and B

    std::cout << "GEMM" << std::endl;
    gemm(0, 0, M, N, K, 1.0, col, K, wei, K, 1.0, col, pd()->OW());
    std::cout << "After GEMM" << std::endl;

    std::cout << "Before col2im" << std::endl;
    // col2im
    col2im_cpu(col, pd()->OC(), pd()->OH(), pd()->OW(), 
        pd()->KH(), pd()->KW(), 1 , 1, 
        1, 1, col);
    std::cout << "After col2im" << std::endl;
    dst = col;
    return status::success;
}

template struct rv64_gemm_convolution_fwd_t<data_type::f32>;


} // rv64
} // cpu
} // impl
} // dnnl