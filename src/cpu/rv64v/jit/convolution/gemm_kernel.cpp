#include <stddef.h>
#include <iostream>
#include <vector>
#include <cstdlib> 
#include <algorithm>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"
#include "common/type_helpers.hpp"
#include "cpu/rv64v/jit/convolution/gemm_kernel.hpp"


namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace gemm{

using namespace rvjit;
using jit_conv_kernel_args_t = convolution_schedule_t::jit_conv_kernel_args_t;


// Implementing DARKNET vectorized functions using JIT instructions
using namespace dnnl::impl::utils;
using namespace rvjit::vtype;
template <typename T>
void jit_convolution_kernel_t::im2col_cpu(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, 
    const T data_im,const T data_col,int channels,  int height,  int width,int ksize,  int stride, int pad)
{
    int c,h,w, index; 
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    // register_pool_t tmp_pool({t0,t1,t2,t3,t4,t5,t6,a7,a6,a5,a4,a3,a2,a1});
        
    const gpr_t gvl = tmp.pick();
    const gpr_t vlen = tmp.pick();
    const gpr_t w_reg = tmp.pick();
    const gpr_t h_offset_reg = tmp.pick();
    const gpr_t intermediate_reg = tmp.pick();
    const gpr_t tmp1 = tmp.pick();
    const gpr_t four = tmp.pick();
    const gpr_t tmp2 = tmp.pick();
    const gpr_t val_reg = tmp.pick();
    const gpr_t src = tmp.pick();
    const gpr_t col = tmp.pick();
    const vr_t wcol = vout[0];
    const vr_t OFFSET = vout[1];
    const vr_t WIDTHCOL = vout[2];
    const vr_t PAD = vout[3];
    const vr_t STRIDE = vout[4];
    const vr_t INTER = vout[5];
    const vr_t intermediate1 = vout[6];
    const vr_t imcol = vout[7];
    const vr_t intermediate2 = vout[8];
    const vr_t colindex = vout[9];
    const vr_t WIDTH = vout[10];
    const vr_t HEIGHT = vout[11];
    const vr_t CIM = vout[12];
    const vr_t imrow = vout[13];
    const vr_t FOUR = vout[14];
    const vr_t XERO = vout[15];
    const vr_t XERO1 = vout[16];
    const vr_t intermediate5 = vout[17];
    const vr_t VAL = vout[18];
    const vr_t dataim = vout[19];
    const vr_t datacol = vout[20];

    load_constant(src, data_im);
    load_constant(col, data_col);
    load_constant(vlen, cfg.vlen);

    vtype_t sew = e32;
    vsetvli(gvl, vlen, sew | vlmul(1));
    const int channels_col = channels * ksize * ksize;
    // initialize channels register
    const gpr_t channel = tmp.pick();
    load_constant(channel, channels_col);
    c = 0;
    if (channels_col > 1)   load_constant(channel, channels_col);
    L("channels");
    
    //for (c = 0; c < channels_col; ++c) {
    int w_offset = c % ksize;
    load_constant(w_reg, w_offset);
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (h = 0; h < height_col; ++h) {
        int im_row = h_offset + h * stride;
        int intermediate = (c * height_col + h);
        load_constant(intermediate_reg, intermediate);
        im_row -= pad;
        int val = width*(im_row + height*c_im);
        load_constant(val_reg, val);
        for (w = 0; w < width_col; w += cfg.vlen) {   
            //Index calculation
            load_constant(tmp1, w);
            vle32(wcol, tmp1); // load
            vmv_sx(OFFSET, w_reg); // broadcast
            load_constant(tmp1, pad);
            vmv_sx(PAD, tmp1); //broadcast
            load_constant(tmp1, stride);
            vmv_sx(STRIDE, tmp1); //broadcast

            vmul_vv(intermediate1, STRIDE, wcol); // multiplication
            vadd_vv(imcol, intermediate1, OFFSET); // addition

            load_constant(tmp1, width_col);
            vmv_sx(WIDTHCOL, tmp1); //broadcast

            vmv_sx(INTER, intermediate_reg); //broadcast
            
            vmul_vv(intermediate2, INTER, WIDTHCOL); // multiplication

            vadd_vv(colindex, intermediate2, wcol); // addition

            vsub_vv(imcol, imcol, PAD); // subtract

            //broadcast for conditional statement
            load_constant(tmp1, width);
            vmv_sx(WIDTH, tmp1); //broadcast

            load_constant(tmp1, height);
            vmv_sx(HEIGHT, tmp1); //broadcasts

            load_constant(tmp1, c_im);
            vmv_sx(CIM, tmp1); //broadcast

            load_constant(tmp1, im_row);
            vmv_sx(imrow, tmp1); //broadcast

            //Broadcast 4 for index calculation (index*4 for float 32bit)
            //int l = 4;
            load_constant(four, 4);
            vmv_sx(FOUR, four); //broadcast

            int z=0;
            float z1=0.0;
            
            load_constant(tmp2, z); 
            vfsub_vv(XERO1, XERO1, XERO1); // set XERO1 to 0.0
            vmv_sx(XERO, tmp2); //broadcast

            //Calculate mask
            vmask_t colmask, colmask1, colmask2;
            vmask_t rowmask, rowmask1, rowmask2;
            vmask_t mask, mask1, mask2, mask3, mask4;

            vmsgt_vx(colmask, imcol, XERO);
            vmslt_vx(colmask1, imcol, WIDTH);
            vmseq_vv(colmask2, imcol, XERO);

            vmsgt_vx(rowmask, imrow, XERO);
            vmslt_vx(rowmask1, imrow, HEIGHT);
            vmseq_vv(rowmask2, imrow, XERO);
            
            vmand_mm(mask, rowmask1, colmask1);
            vmor_mm(mask1, colmask, colmask2);
            vmor_mm(mask2, rowmask, rowmask2);
            vmand_mm(mask3, mask1, mask2);
            vmand_mm(mask4, mask, mask3);

            //Calculate val+imcol for final index
            vmv_vx(intermediate5, val_reg);
            vadd_vv(VAL, imcol, intermediate5, mask4);

            //Index multiply with 4
            vmul_vv(VAL, VAL, FOUR);
            vmul_vv(colindex, colindex, FOUR);

            //vload with indexed mask
            vlox(dataim, src, VAL, mask4);
            //store with index
            vsox(datacol, col, colindex, mask4);
            //w += cfg.vlen;//gvl;
            
            }

    }
    if (channel > 1) {
        c++;
        addi(channel, channel, -1);
        bnez(channel, "channels");
    }
}


/***********************3. loop interchange with manual vectorization with ALPHA!=1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling with unroll degree 24*/

void jit_convolution_kernel_t::gemm_nn_unroll16(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,
    int ii, int jj, int kk, const size_t A, const size_t B, const size_t C, float ALPHA, int M, int N, 
    int K,  int lda,int ldb,int ldc)
{
    std::cout << "Entered JIT gemm_nn_unroll16" << std::endl;
    const gpr_t gvl = tmp.pick();
    const gpr_t vlen = tmp.pick();

    const gpr_t a_index = tmp.pick();
    const gpr_t b_index = tmp.pick();
    const gpr_t c_index = tmp.pick();
    const gpr_t a_index1 = tmp.pick();
    const gpr_t a_index2 = tmp.pick();
    const gpr_t a_index3 = tmp.pick();

    const vr_t vc = vout[0], vc1 = vout[1], vc2 = vout[2], vc3 = vout[3];
    const vr_t vc4 = vout[4], vc5 = vout[5], vc6 = vout[6], vc7 = vout[7];
    const vr_t vc8 = vout[8], vc9 = vout[9], vc10 = vout[10], vc11 = vout[11];
    const vr_t vc12 = vout[12], vc13 = vout[13], vc14 = vout[14], vc15 = vout[15];
    const vr_t vaalpha = vout[16], vaalpha1 = vout[17], vaalpha2 = vout[18];
    const vr_t vaalpha3 = vout[19], vaalpha4 = vout[20], vaalpha5 = vout[21];
    const vr_t vaalpha6 = vout[22], vaalpha7 = vout[23], vaalpha8 = vout[24];
    const vr_t vaalpha9 = vout[25], vaalpha10 = vout[26], vaalpha11 = vout[27];
    const vr_t vaalpha12 = vout[28], vaalpha13 = vout[29], vaalpha14 = vout[30];
    const vr_t vaalpha15 = vout[31];
    vr_t vb;

    vtype_t sew = e32;
    vsetvli(gvl, vlen, sew | vlmul(1)); 
    int i1=ii, j1=jj, k1=kk;
    int i=0,j=0,k=0;
    for ( j = 0; j < N; j += cfg.vlen) {
    for (i = 0; i < M-15; i += 16) {

        load_constant(c_index, C+((i+i1)*ldc+(j+j1)*sizeof(float)));
        vle32(vc, c_index);

        load_constant(c_index, C+((i+i1+1)*ldc+(j+j1)*sizeof(float)));
        vle32(vc1, c_index);

        load_constant(c_index, C+((i+i1+2)*ldc+(j+j1)*sizeof(float)));
        vle32(vc2, c_index);

        load_constant(c_index, C+((i+i1+3)*ldc+(j+j1)*sizeof(float)));
        vle32(vc3, c_index);

        load_constant(c_index, C+((i+i1+4)*ldc+(j+j1)*sizeof(float)));
        vle32(vc4, c_index);

        load_constant(c_index, C+((i+i1+5)*ldc+(j+j1)*sizeof(float)));
        vle32(vc5, c_index);

        load_constant(c_index, C+((i+i1+6)*ldc+(j+j1)*sizeof(float)));
        vle32(vc6, c_index);

        load_constant(c_index, C+((i+i1+7)*ldc+(j+j1)*sizeof(float)));
        vle32(vc7, c_index);

        load_constant(c_index, C+((i+i1+8)*ldc+(j+j1)*sizeof(float)));
        vle32(vc8, c_index);

        load_constant(c_index, C+((i+i1+9)*ldc+(j+j1)*sizeof(float)));
        vle32(vc9, c_index);

        load_constant(c_index, C+((i+i1+10)*ldc+(j+j1)*sizeof(float)));
        vle32(vc10, c_index);

        load_constant(c_index, C+((i+i1+11)*ldc+(j+j1)*sizeof(float)));
        vle32(vc11, c_index);

        load_constant(c_index, C+((i+i1+12)*ldc+(j+j1)*sizeof(float)));
        vle32(vc12, c_index);

        load_constant(c_index, C+((i+i1+13)*ldc+(j+j1)*sizeof(float)));
        vle32(vc13, c_index);

        load_constant(c_index, C+((i+i1+14)*ldc+(j+j1)*sizeof(float)));
        vle32(vc14, c_index);

        load_constant(c_index, C+((i+i1+15)*ldc+(j+j1)*sizeof(float)));
        vle32(vc15, c_index);
        for ( k = 0; k < K; k ++) {
            vb = vout[31]; // Using last register for B

            load_constant(b_index, B+(((k+(K*(j/ldb)))*ldb)+0)*sizeof(float));
            vle32(vb, b_index);
            load_constant(a_index, A+(i+lda*k)*sizeof(float));
            vle32(vaalpha, a_index);
            vfmacc_vv(vc, vaalpha, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+1)+lda*k)*sizeof(float));
            vle32(vaalpha1, a_index);
            vfmacc_vv(vc1, vaalpha1, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+2)+lda*k)*sizeof(float));
            vle32(vaalpha2, a_index);
            vfmacc_vv(vc2, vaalpha2, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+3)+lda*k)*sizeof(float));
            vle32(vaalpha3, a_index);
            vfmacc_vv(vc3, vaalpha3, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+4)+lda*k)*sizeof(float));
            vle32(vaalpha4, a_index);
            vfmacc_vv(vc4, vaalpha4, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+5)+lda*k)*sizeof(float));
            vle32(vaalpha5, a_index);
            vfmacc_vv(vc5, vaalpha5, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+6)+lda*k)*sizeof(float));
            vle32(vaalpha6, a_index);
            vfmacc_vv(vc6, vaalpha6, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+7)+lda*k)*sizeof(float));
            vle32(vaalpha7, a_index);
            vfmacc_vv(vc7, vaalpha7, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+8)+lda*k)*sizeof(float));
            vle32(vaalpha8, a_index);
            vfmacc_vv(vc8, vaalpha8, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+9)+lda*k)*sizeof(float));
            vle32(vaalpha9, a_index);
            vfmacc_vv(vc9, vaalpha9, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+10)+lda*k)*sizeof(float));
            vle32(vaalpha10, a_index);
            vfmacc_vv(vc10, vaalpha10, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+11)+lda*k)*sizeof(float));
            vle32(vaalpha11, a_index);
            vfmacc_vv(vc11, vaalpha11, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+12)+lda*k)*sizeof(float));
            vle32(vaalpha12, a_index);
            vfmacc_vv(vc12, vaalpha12, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+13)+lda*k)*sizeof(float));
            vle32(vaalpha13, a_index);
            vfmacc_vv(vc13, vaalpha13, vb); // sum += ALPHA*A*B
            load_constant(a_index, A+((i+14)+lda*k)*sizeof(float));
            vle32(vaalpha14, a_index);
            vfmacc_vv(vc14, vaalpha14, vb); // sum += ALPHA*A*B
            vb = vout[16]; // Switch to 16th register for B due to register amount
            load_constant(b_index, B+(((k+(K*(j/ldb)))*ldb)+0)*sizeof(float));
            vle32(vb, b_index);
            load_constant(a_index, A+((i+15)+lda*k)*sizeof(float));
            vle32(vaalpha15, a_index);
            vfmacc_vv(vc15, vaalpha15, vb); // sum += ALPHA*A*B  
            
        } 
        load_constant(c_index, C+((i+i1)*ldc+(j+j1)*sizeof(float)));
        vse32(vc, c_index);
        load_constant(c_index, C+((i+i1+1)*ldc+(j+j1)*sizeof(float)));
        vse32(vc1, c_index);
        load_constant(c_index, C+((i+i1+2)*ldc+(j+j1)*sizeof(float)));
        vse32(vc2, c_index);
        load_constant(c_index, C+((i+i1+3)*ldc+(j+j1)*sizeof(float)));
        vse32(vc3, c_index);
        load_constant(c_index, C+((i+i1+4)*ldc+(j+j1)*sizeof(float)));
        vse32(vc4, c_index);
        load_constant(c_index, C+((i+i1+5)*ldc+(j+j1)*sizeof(float)));
        vse32(vc5, c_index);
        load_constant(c_index, C+((i+i1+6)*ldc+(j+j1)*sizeof(float)));
        vse32(vc6, c_index);
        load_constant(c_index, C+((i+i1+7)*ldc+(j+j1)*sizeof(float)));
        vse32(vc7, c_index);
        load_constant(c_index, C+((i+i1+8)*ldc+(j+j1)*sizeof(float)));
        vse32(vc8, c_index);
        load_constant(c_index, C+((i+i1+9)*ldc+(j+j1)*sizeof(float)));
        vse32(vc9, c_index);
        load_constant(c_index, C+((i+i1+10)*ldc+(j+j1)*sizeof(float)));
        vse32(vc10, c_index);
        load_constant(c_index, C+((i+i1+11)*ldc+(j+j1)*sizeof(float)));
        vse32(vc11, c_index);
        load_constant(c_index, C+((i+i1+12)*ldc+(j+j1)*sizeof(float)));
        vse32(vc12, c_index);
        load_constant(c_index, C+((i+i1+13)*ldc+(j+j1)*sizeof(float)));
        vse32(vc13, c_index);
        load_constant(c_index, C+((i+i1+14)*ldc+(j+j1)*sizeof(float)));
        vse32(vc14, c_index);
        load_constant(c_index, C+((i+i1+15)*ldc+(j+j1)*sizeof(float)));
        vse32(vc15, c_index);
    }
    
    }

    int i_left=i;
    //itr=0;
    for (int j = 0; j < N; ) {
        for (i=i_left; i < M; i += 4) {    // change according to unroll degree
        load_constant(c_index, C+((i+i1)*ldc+(j+j1)*sizeof(float)));
        vle32(vc, c_index);
        if(i + 1 < M) {
            load_constant(c_index, C+((i+i1+1)*ldc+(j+j1)*sizeof(float)));
            vle32(vc1, c_index);
        }
        if(i + 2 < M) {
            load_constant(c_index, C+((i+i1+2)*ldc+(j+j1)*sizeof(float)));
            vle32(vc2, c_index);
        }
        if(i + 3 < M) {
            load_constant(c_index, C+((i+i1+3)*ldc+(j+j1)*sizeof(float)));
            vle32(vc3, c_index);
        }
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < K; k ++) {
                load_constant(a_index, A+(i+lda*k)*sizeof(float));
                if (i+1 < M) {load_constant(a_index1, A+(i+1+lda*k)*sizeof(float));}
                if (i+2 < M) {load_constant(a_index2, A+(i+2+lda*k)*sizeof(float));}
                if (i+3 < M) {load_constant(a_index3, A+(i+3+lda*k)*sizeof(float));}

                vle32(vaalpha, a_index);
                if (i+1 < M) {vle32(vaalpha1, a_index1);}
                if (i+2 < M) {vle32(vaalpha2, a_index2);} // ALPHA*A
                if (i+3 < M) {vle32(vaalpha3, a_index3);} // ALPHA*A

                load_constant(b_index, B+(((k+(K*(j/ldb)))*ldb)+0)*sizeof(float));
                vle32(vb, b_index);
                vfmacc_vv(vc, vaalpha, vb);
                if (i+1 < M) {vfmacc_vv(vc1, vaalpha1, vb);} // sum += ALPHA*A*B
                if (i+2 < M) {vfmacc_vv(vc2, vaalpha2, vb);} // sum += ALPHA*A*B
                if (i+3 < M) {vfmacc_vv(vc3, vaalpha3, vb);}// sum += ALPHA*A*B
            }
            load_constant(c_index, C+((i+i1)*ldc+(j+j1)*sizeof(float)));
            vse32(vc, c_index);
            if (i+1 < M) {load_constant(c_index, C+((i+i1+1)*ldc+(j+j1)*sizeof(float))); vse32(vc1, c_index);}
            if (i+2 < M) {load_constant(c_index, C+((i+i1+2)*ldc+(j+j1)*sizeof(float))); vse32(vc2, c_index);}
            if (i+3 < M) {load_constant(c_index, C+((i+i1+3)*ldc+(j+j1)*sizeof(float))); vse32(vc3, c_index);}
        }
        j += cfg.vlen;
  }
}


//6-loops with packA and PackB
void jit_convolution_kernel_t::gemm_nn_pack2(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,int M, int N, int K, float ALPHA,
        size_t A, int lda,
        size_t B, int ldb,
        size_t C,  int ldc, int BlockM, int BlockN, int BlockK, size_t transposeB, size_t transposeA)
    {        
    int ii,jj,kk,i,j,k;
    //int ld = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);//16;
    const gpr_t ld = tmp.pick();
    const gpr_t vlen = tmp.pick();
    load_constant(vlen, cfg.vlen);
    //vsetvli(ld, vlen, e32 | vlmul(1));
    long gvl;
    const vr_t tmp_reg = vout[0];
    const gpr_t b_index = tmp.pick();
    const gpr_t bT_index = tmp.pick();
    const gpr_t a_index = tmp.pick();
    const gpr_t aT_index = tmp.pick();
    gvl = cfg.vlen;

    for (jj = 0; jj < N; jj+=BlockN) {
        int Nc = ((jj+BlockN>N)?(N-jj):(BlockN));
        for (kk = 0; kk < K; kk+=BlockK) {
            int Kc = ((kk+BlockK > K)?(K-kk):(BlockK));
            int itr=0;
            for(int j=0;j<Nc;){
                for(int k=0;k<Kc;k++){

                    //      transposeB[k*Kc+j] = B[(k+kk)*ldb+(j+jj)];
                    //__epi_2xf32 tmp = __builtin_epi_vload_2xf32( &B[(k+kk)*ldb+(j+jj)], gvl);
                    load_constant(b_index, B+(k+kk)*ldb+(j+jj)*sizeof(float));
                    vle32(tmp_reg, b_index);
                    //svst1(pg, &transposeB[((k+(Kc*itr))*ld)+0], tmp);
                    //__builtin_epi_vstore_2xf32( &transposeB[((k+(Kc*(j/ld)))*ld)+0], tmp, gvl);
                    load_constant(bT_index, (((k+(Kc*(j/ld)))*ld)+0)*sizeof(float));
                    vse32(tmp_reg, bT_index);
                    //transposeB[k*Nc+j] = B[(k+kk)*ldb+(j+jj)];
                }
                itr++;
                j+=gvl;
            }
            for (ii = 0; ii < M; ii+=BlockM) {
            int Mc = ((ii+BlockM >M)?(M-ii):(BlockM)) ;

            int itr1=0;
            for(int i=0;i<Mc;i++){
                for(int k=0;k<Kc;k++){
                    //      	__epi_2xf32 tmp = __builtin_epi_vload_2xf32(&A[(i+ii)*lda+(k+kk)], gvl);
                    //    	__builtin_epi_vstore_strided_2xf32(&transposeA[k*Mc+i], tmp, Mc*4, gvl);
                    //transposeA[k*Mc+i] = A[(i+ii)*lda+(k+kk)];
                    load_constant(a_index, A+((i+ii)*lda+(k+kk))*sizeof(float));
                    load_constant(aT_index, (k*Mc+i)*sizeof(float));
                    vle32(tmp_reg, a_index);
                    vse32(tmp_reg, aT_index);
                    }
                }
                gemm_nn_unroll16(vout, nvregs, tmp, ii,jj,kk,transposeA,transposeB, C,ALPHA, Mc,Nc, Kc, Mc,ld,ldc);
            }
        }
    }
}

void jit_convolution_kernel_t::gemm_cpu(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,
        int TA, int TB, int M, int N, int K, float ALPHA, 
        size_t A, int lda, 
        size_t B, int ldb,
        float BETA,
        size_t C, int ldc)
{
    std::cout << "Enter GEMM CPU" << std::endl;
    if(!TA && !TB)
    {
	    /*** enable below for the 6-loops packed implementations */
        int blockM = std::min(16*2, M);
        int blockN = std::min(512*2, N);
        int blockK = std::min(128*2, K);

        // Using new to allocate memory
        float* transposeB = new (std::nothrow) float[blockM * blockN * blockK];
        float* transposeA = new (std::nothrow) float[blockM * blockN * blockK];

        if (!transposeB) {
            std::cerr << "Fatal: failed to allocate bytes for transposeB.\n";
            exit(EXIT_FAILURE); // Prefer EXIT_FAILURE or EXIT_SUCCESS
        }

        if (!transposeA) {
            std::cerr << "Fatal: failed to allocate bytes for transposeA.\n";
            exit(EXIT_FAILURE);
        }
        size_t transposeB_ = reinterpret_cast<size_t>(transposeB);
        size_t transposeA_ = reinterpret_cast<size_t>(transposeA);
        //gemm_nn_original(M,N,K,ALPHA,A, lda,B, ldb, C, ldc);	
        std::cout << "Calling gemm_nn_pack2" << std::endl;
        gemm_nn_pack2(vout, nvregs, tmp, M, N, K, ALPHA,A, lda, B, ldb,C, ldc, blockM, blockN, blockK, transposeB_, transposeA_);
        delete[] transposeB;
        transposeB = nullptr; // Using nullptr

        delete[] transposeA;
        transposeA = nullptr; // Using nullptr

    /*** 3-loop implementation */
	//gemm_nn_unroll16(vout, nvregs, tmp, M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    }/*
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);*/
}

void jit_convolution_kernel_t::code(convolution_schedule_t::jit_conv_kernel_args_t kargs){
    
    const int oh = cfg.oh; // output height
    const int ow = cfg.ow; // output width
    const int ih = cfg.ih; // input height
    const int iw = cfg.iw; // input width
    const int kh = cfg.kh; // kernel height
    const int kw = cfg.kw; // kernel width
    const int oc = cfg.oc; // output channels
    const int ic = cfg.ic; // input channels
    const int stride_h = cfg.stride_h; // stride height
    const int stride_w = cfg.stride_w; // stride width
    const int l_pad = cfg.l_pad; // left padding
    const int t_pad = cfg.t_pad; // top padding
    const int vlen = cfg.vlen; // vector length
    const int nvregs = traits.erbw * traits.erbc;
    const size_t wei_sew = types::data_type_size(cfg.wei_dt);
    const size_t bia_sew = cfg.with_bias ? types::data_type_size(cfg.bias_dt) : 0;
    const size_t src_sew = types::data_type_size(cfg.src_dt);
    const size_t dst_sew = types::data_type_size(cfg.dst_dt);
    
    // Offset to output pointer field in kernel args structure
    //const auto args_dst_ptr = offsetof(jit_conv_kernel_args_t, dst);
    //const auto args_src_ptr = offsetof(jit_conv_kernel_args_t, src);
    size_t src_addr = reinterpret_cast<size_t>(kargs.src);
    size_t dst_addr = reinterpret_cast<size_t>(kargs.dst);
    size_t wei_addr = reinterpret_cast<size_t>(kargs.wei);
    int M = oc; // Number of output channels
    int N = oh * ow; // Output spatial dimensions (flattened)
    int K = ic * kh * kw; // Dimension shared by A and B

    /// Output register block
    vr_t vout[32];
    for (int i = 0; i < nvregs; ++i)
        vout[i] = static_cast<vr_t>(i);
    /// Pool of available caller-saved general purpose registers
    register_pool_t tmp_pool({t0,t1,t2,t3,t4,t5,t6,a7,a6,a5,a4,a3,a2,a1});
    im2col_cpu(vout, nvregs, tmp_pool, src_addr, dst_addr, ic, ih, iw, kh, stride_h, l_pad);
    gemm_cpu(vout, nvregs, tmp_pool, 0, 0, M, N, K, 1.0, wei_addr, K, dst_addr, N, 0.0, dst_addr, N);
}

/*
void jit_convolution_kernel_t::code() {
    
    const int nvregs = traits.erbw * traits.erbc;
    const size_t wei_sew = types::data_type_size(cfg.wei_dt);
    const size_t bia_sew = cfg.with_bias ? types::data_type_size(cfg.bias_dt) : 0;
    const size_t src_sew = types::data_type_size(cfg.src_dt);
    const size_t dst_sew = types::data_type_size(cfg.dst_dt);

    const size_t out_sew =
        utils::pick_by_prop_kind(cfg.prop_kind, dst_sew, src_sew, wei_sew);

    /// Offset to output pointer field in kernel args structure
    const size_t args_out_ptr = utils::pick_by_prop_kind(cfg.prop_kind,
                                offsetof(jit_conv_kernel_args_t, dst),
                                offsetof(jit_conv_kernel_args_t, src),
                                offsetof(jit_conv_kernel_args_t, wei));
    /// Output tensor W dimension stride
    const size_t w_off = utils::pick_by_prop_kind(cfg.prop_kind,
                                dst_sew * cfg.ocb,
                                src_sew * cfg.icb * cfg.stride_w,
                                wei_sew * cfg.w_ocb * cfg.w_icb);
    /// Output tensor C inter-block dimension stride
    const size_t c_off = utils::pick_by_prop_kind(cfg.prop_kind,
                                dst_sew * cfg.ocb * cfg.oh * cfg.ow,
                                src_sew * cfg.icb * cfg.ih * cfg.iw,
                                wei_sew * cfg.vlen);
    /// Output register block
    vr_t vout[32];
    for (int i = 0; i < nvregs; ++i)
        vout[i] = static_cast<vr_t>(i);
    /// Pool of available caller-saved general purpose registers
    register_pool_t tmp_pool({t0,t1,t2,t3,t4,t5,t6,a7,a6,a5,a4,a3,a2,a1});

    /// Move data from/to output vector registers and tensor memory region
    const auto move_outputs = [&](bool is_load, register_pool_t pool) {
        const gpr_t out = pool.pick();
        assembly_constant_t asm_c_off;
        assembly_constant_t asm_w_off;


        asm_w_off = asm_const(pool, w_off);
        // Subtract the accumulated W offset
        asm_c_off = asm_const(pool, c_off - (cfg.rbw-1) * w_off);

        ld(out, a0, args_out_ptr);
        if (cfg.rbc > 1)
            prepare_constant(asm_c_off);
        if (cfg.rbw > 1)
            prepare_constant(asm_w_off);

        int w = 0, c = 0;
        bool is_done = false;

        do {
            const auto id = w * cfg.rbc + c;
            if (is_load)
                vl(vout[id], out, out_sew);
            else
                vs(vout[id], out, out_sew);
            //if (!is_bwdw) {
            is_done = utils::nd_iterator_step(c, cfg.rbc, w, cfg.rbw);
            if (!is_done) {
                if (w)
                    add_constant(out, out, asm_w_off);
                else
                    add_constant(out, out, asm_c_off);
            }
        } while(!is_done);
    };

    // Initialization Segment
    do {
        const int sew = utils::pick_by_prop_kind(cfg.prop_kind,
            dst_sew, src_sew, wei_sew);
        const gpr_t vlen = tmp_pool.pick();
        const gpr_t load_partials = tmp_pool.pick();
        ld(vlen, a0, offsetof(jit_conv_kernel_args_t, vlen));
        vsetvli(x0, vlen, vsew(sew) | vlmul(1));
        lw(load_partials, a0, offsetof(jit_conv_kernel_args_t, load_partials));
        bnez(load_partials, "load_psum");
        for (int i = 0; i < nvregs; ++i)
            vxor_vv(vout[i], vout[i], vout[i]);
        j("compute");
        L("load_psum");
        move_outputs(true, tmp_pool);
    } while (0);

    // Compute Segment
    tmp_pool.reset();
    L("compute");
    switch (cfg.prop_kind) {
        case prop_kind::forward: {
            fwdd_inner_loops(vout, nvregs, tmp_pool);
            break;
        }
        case prop_kind::forward_inference: {
            fwdd_inner_loops(vout, nvregs, tmp_pool);
            break;
        }
        default:
            assert(!"unsupported propagation kind");
    }
        
    // Store Partial Sums Segment
    // TODO: this is temporary BIAS support
    if (cfg.with_bias) {//&& is_fwdd) {
        auto tmp   = tmp_pool.pick();
        auto bias_ptr  = tmp_pool.head();
        auto vbias = static_cast<vr_t>(nvregs);

        ld(tmp, a0, offsetof(jit_conv_kernel_args_t, load_partials));
        bnez(tmp, "store_psum");
        ld(bias_ptr, a0, offsetof(jit_conv_kernel_args_t, bias));
        vl(vbias, bias_ptr, bia_sew);
        for (int i = 0; i < nvregs; ++i)
            vfadd_vv(vout[i], vout[i], vbias);
    }
    tmp_pool.reset();
    L("store_psum");
    move_outputs(false, tmp_pool);
    ret();

}
*/


int
starting_point(jit_convolution_configuration_t cfg, int rbw, int kh, int kw) {
    const int dst_sew = types::data_type_size(cfg.dst_dt);
    const int src_sew = types::data_type_size(cfg.src_dt);
    switch (cfg.prop_kind) {
        case prop_kind::forward_inference:
        case prop_kind::forward_training:
            return (kh * (cfg.dilate_h+1) * cfg.iw
                 + kw * (cfg.dilate_w+1) + rbw * cfg.stride_w) * cfg.icb * src_sew;
        case prop_kind::backward_data:
            return (kh * cfg.ow + kw + rbw) * cfg.ocb * dst_sew;
        default:
            assert(false);
            return 0;
    }
}

void jit_convolution_kernel_t::fwdd_inner_loops(rvjit::vr_t *vout, int rb_sz, register_pool_t &rp) {
    // --------------- Number of iterations for compute loops -----------------
    // The following constitutes the micro-kernel compute loops
    // for (icb = 0; icb < min(icb, k_c) / icb; ++icb)
    //  for (wic = 0; wic < min(icb, k_c) / w_icb; ++wic) // only pointwise
    //   for (kh = padTop; kh < k_h - padBot; ++kh)
    //    for (kw = 0; kh < k_w; ++kw)
    //     for (ic = 0; ic < w_icb; ++ic)
    auto const xcb = cfg.icb;
    auto const wxcb = cfg.w_icb;
    auto const kxcb_loop_sz = nstl::max(1, cfg.k_c/xcb);
    auto const xcb_loop_sz  = (cfg.k_c > xcb ? xcb : cfg.k_c) / wxcb;
    const int wei_sew = types::data_type_size(cfg.wei_dt);
    const int src_sew = types::data_type_size(cfg.src_dt); 

    // ---------------------------- Tensor Offsets ----------------------------
    const int kw_off   = cfg.w_icb * cfg.w_ocb * wei_sew;
    const int xw_off   = cfg.stride_w * xcb    * src_sew;
    const int wxcb_off = wxcb                  * src_sew;
    int xcb_off        = cfg.ih * cfg.iw * xcb * src_sew;

    // ---------------------- Code generation conditions ----------------------
    /// A pointwise kernel does not need two loops to cover the ICB block
    const bool c_is_pointwise = cfg.k_h * cfg.k_w == 1;
    /// Generate w_icb loop as icb > w_icb
    const bool c_gen_icb_loop = xcb_loop_sz > 1 && !c_is_pointwise;
    /// Walk the register block applying immediate offsets (save on adds)
    const bool c_use_imm_rbw = imm_range / 2 > xw_off;

    // ------------------------------ Registers -------------------------------
    const gpr_t vsrc = rp.pick();
    const gpr_t xsrc = rp.pick();
    const gpr_t ptr_bcast = rp.pick();  // Working pointer to src tensor
    const gpr_t xc_off = rp.pick();     // Accumulated ic offset within IC blk
    const gpr_t xc_off_max = rp.pick(); // Max ic offset within IC blk (use in 'beq`)
    const gpr_t i_kxc = rp.pick();      // k_ic loop iterator (sub-tensors)
    const gpr_t i_xcb = rp.pick();      // icb loop iterator (number of w_icb blocks)
    auto wei_off = asm_const(rp, cfg.vlen * wei_sew);
    auto rbw_off = asm_const(rp, traits.erbw == 1 ? 0 :
        c_use_imm_rbw ? imm_range : xw_off);
    const gpr_t rbw_start = rp.pick();  // Pointer to the first activation
    const gpr_t tmp = rp.pick();        // Multi-purpose temporary across the ukernel

    /// VFMA source tensor scalar operand (implicit vector broadcast)
    const fpr_t f_bcast[4] = {ft0, ft1, ft2, ft3};
    static constexpr unsigned int nf_bcast = 4;
    unsigned int f_bcast_id = 0;
    /// VFMA weights tensor vector operand
    const vr_t vwei = static_cast<vr_t>(rb_sz);

    // --------------------- Control variables and lambdas --------------------
    /// Update the src pointer to the next convolution window (out register)
    /// @return true when the update requires an add instruction
    /// @details If the rbw_off is an immediate, the function only returns true
    /// when the accumulated offset overflows the 12bit representation range.
    auto should_issue_add_to_update_rbw_offset = [&](int &off) {
        if (c_use_imm_rbw) {
            off += xw_off;
            if (can_be_imm12(off))
                return false;
            off -= imm_range;
        }
        return true;
    };

    // -------------------------- Compute kernel begin ------------------------
    ld(vsrc, a0, offsetof(jit_conv_kernel_args_t, wei));
    ld(xsrc, a0, offsetof(jit_conv_kernel_args_t, src));
    prepare_constant(wei_off);
    load_constant(xc_off_max, (c_is_pointwise ? xcb_loop_sz : 1) * wxcb_off);
    if (traits.erbw > 1)    prepare_constant(rbw_off);
    if (kxcb_loop_sz > 1)   load_constant(i_kxc, kxcb_loop_sz);
    L("kxc_loop");
    if (c_gen_icb_loop)     load_constant(i_xcb, xcb_loop_sz);
    L("xcb_loop");

    int wei_skip = traits.rbpadT * cfg.kw;
    for (int kh = traits.rbpadT; kh < cfg.k_h - traits.rbpadB; ++kh) {
        for (int kw = 0; kw < cfg.k_w; ++kw) {
            /// The first vout index this loop after padding mask
            const int rw_s = traits.rbpadL > kw ? traits.rbpadL - kw : 0;
            /// The last vout index this loop after padding mask
            const int rw_e = traits.erbw - (traits.rbpadR >= cfg.k_w-kw
                ? (traits.rbpadR - (cfg.k_w - (kw+1))) : 0);

            if (rw_e - rw_s < 1) {
                ++wei_skip;
                continue;
            } else if (wei_skip) {
                // Adjust the weight ptr to account for padding
                add_constant(vsrc, vsrc, tmp, wei_skip * kw_off);
                wei_skip = 0;
            }

            // Reset the intra-block channel offset, with range: [0,w_icb)
            li(xc_off, 0);

            // Adjust the working source pointer to the first activation point
            /// The offset from the start of the register the block
            int rbw_imm = (c_use_imm_rbw && traits.erbw > 1) ? imm_min() : 0;
            /// The offset for the starting point of the first output register
            int spatial_offset = starting_point(cfg, rw_s, kh, kw);
            add_constant(rbw_start, xsrc, tmp, spatial_offset - rbw_imm);

            // Start the inner-most loop over w_icb for this point (kh,kw)
            char local_label[16]; /// The current label name
            snprintf(local_label, 16, "l%d", kh * cfg.k_w + kw);
            L(local_label);

            // Set the src ptr to the IC offset at the current sub-tensor
            add(ptr_bcast, rbw_start, xc_off);

            // Load the next vector operand
            vl(vwei, vsrc, wei_sew);
            add_constant(vsrc, vsrc, wei_off);

            // Reuse the vector operand across the output register block
            flw(f_bcast[f_bcast_id], ptr_bcast, rbw_imm);
            vfmacc_vf(vout[rw_s], f_bcast[f_bcast_id], vwei);
            f_bcast_id = (f_bcast_id + 1) % nf_bcast;
            for (int rw = rw_s+1; rw < rw_e; rw++) {
                if (should_issue_add_to_update_rbw_offset(rbw_imm))
                    add_constant(ptr_bcast, ptr_bcast, rbw_off);
                flw(f_bcast[f_bcast_id], ptr_bcast, rbw_imm);
                vfmacc_vf(vout[rw], f_bcast[f_bcast_id], vwei);
                f_bcast_id = (f_bcast_id + 1) % nf_bcast;
            }

            // Loop end: one iteration over the non-vectorized dim block
            addi(xc_off, xc_off, src_sew);
            bne(xc_off, xc_off_max, local_label);
        }
        wei_skip += cfg.kw - cfg.k_w;
    }
    wei_skip += (cfg.kh + traits.rbpadB - cfg.k_h) * cfg.kw;

    if (c_gen_icb_loop || kxcb_loop_sz > 1) {
        // Adjust the weight ptr to account for masked computations due to pad
        if (wei_skip) {
            add_constant(vsrc, vsrc, tmp, wei_skip * kw_off);
            wei_skip = 0;
        }
        // Adjust the src pointer to account for computed channels 
        add(xsrc, xsrc, xc_off_max);
        xcb_off -= (xcb_loop_sz * wxcb_off); // remove accum increments to src
    }

    // Loop end: completed a w_icb segment of the IC block
    if (c_gen_icb_loop) {
        addi(i_xcb, i_xcb, -1);
        bnez(i_xcb, "xcb_loop");
    }

    // Loop end: computed the IC block, go to next sub-tensor
    if (kxcb_loop_sz > 1) {
        add_constant(xsrc, xsrc, tmp, xcb_off);
        addi(i_kxc, i_kxc, -1);
        bnez(i_kxc, "kxc_loop");
    }
}


} // namespace gemm
} // namespace dnnl
} // namespace impl
} // namespace cpu
} // namespace riscvv