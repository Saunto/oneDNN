#include "cpu/rv64v/rv64_gemm_convolution.hpp"
#include "common/utils.hpp"
#include "cpu/gemm_convolution_utils.hpp"
    
#include <cstdlib>
#include <iostream>
#include <vector>
//#include "riscv_vector.h"
#include "/home/samjons/thesis/llvm-EPI-development-toolchain-cross/lib/clang/19/include/riscv_vector.h"
namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// im2col and col2im comes from Berkeley Vision's Caffe
// https://github.com/BVLC/caffe/blob/master/LICENSE
float im2col_get_pixel(const float* im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}
// Function to check if 'a' is greater than or equal to zero and less than 'b'
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename T>
void im2col_cpu(const T* data_im, int channels, int height, int width,
            int ksize, int stride, int pad, T* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    const int size = channels_col * height_col * width_col;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}

void col2im_add_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}

///*
template <typename T>
void col2im_cpu(T* data_col,
         int channels,  int height,  int width,
         int ksize,  int stride, int pad, T* data_im) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad, val);
            }
        }
    }
                
}//*/

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
            float A_PART = ALPHA*A[i*lda+k];
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
            float sum = 0;
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
            float A_PART = ALPHA*A[k*lda+i];
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
            float sum = 0;
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

    int size = pd()->KH() * pd()->KW() * pd()->IC() * pd()->OH() * pd()->OW();
    std::cout << "********************************************" << std::endl;
    /*
    std::cout << "Kernel Height: " << pd()->KH() << std::endl;
    std::cout << "Kernel Width: " << pd()->KW() << std::endl;
    std::cout << "Input Channels: " << pd()->IC() << std::endl;
    std::cout << "Input Height: " << pd()->IH() << std::endl;
    std::cout << "Input Width: " << pd()->IW() << std::endl;
    std::cout << "Output Channels: " << pd()->OC() << std::endl;
    std::cout << "Output Height: " << pd()->OH() << std::endl;
    std::cout << "Output Width: " << pd()->OW() << std::endl;

    std::cout << "Allocated size of col: " << size << std::endl;
    std::cout << "Not allocated size of src: " << src_mb_size << std::endl;
    std::cout << "Allocated size of dst: " << dst_mb_size << std::endl;
    */
    std::cout << "--------------------------------------------" << std::endl;

    // Allocate memory for the intermediate data
    // Using float col[size]; does not work due to stack size limitations
    // Using float* col = new float[size]; does work since it uses the heap  
    
    float* col = new float[size]; 
    float* _dst = new float[size];// Temporary memory for the output tensor
    float* _gemm = new float[size];
    // im2col
    std::cout << "Running im2col" << std::endl;
    im2col_cpu(src, pd()->IC(), pd()->IH(), pd()->IW(), 
        pd()->KH(), 1, 1, col);
    // GEMM: A * B = C
    // A is the input tensor, B is the weight tensor, and C is the output tensor
    std::cout << "--------------------------------------------" << std::endl;

     // Calculate dimensions for GEMM based on convolution parameters
    int M = pd()->OC(); // Number of output channels
    int N = pd()->OH() * pd()->OW(); // Output spatial dimensions (flattened)
    int K = pd()->IC() * pd()->KH() * pd()->KW(); // Dimension shared by A and B

    std::cout << "Running GEMM" << std::endl;
    gemm(0, 0, M, N, K, 1.0, col, K, wei, K, 1.0, _gemm, pd()->OW());
    std::cout << "--------------------------------------------" << std::endl;
    

    // col2im
    std::cout << "Running col2im" << std::endl;
    col2im_cpu(_gemm, pd()->IC(), pd()->OH(), pd()->OW(), 
        pd()->KH(), 1, 1, dst);
    std::cout << "********************************************" << std::endl;

    //std::cout << "Copying col to dst" << std::endl;
    // Check size of dst and _dst
    //size_t dst_size = sizeof(data_t) * dst_mb_size;
    //size_t _dst_size = sizeof(float) * dst_mb_size;

    //std::cout << "Size of dst: " << dst_size << std::endl;
    //std::cout << "Size of _dst: " << _dst_size << std::endl;
    //std::memcpy(dst, _dst, sizeof(data_t) * dst_mb_size); // Ensure correct size

    delete[] _dst;
    delete[] _gemm;
    delete[] col;

    return status::success;
}

template struct rv64_gemm_convolution_fwd_t<data_type::f32>;


} // rv64
} // cpu
} // impl
} // dnnl