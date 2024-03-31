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

void jit_convolution_kernel_t::im2col_cpu(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, 
    int channels,  int height,  int width,int ksize,  int stride, int pad)
{
    int h,w; 
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    //register_pool_t tmp_pool({t0,t1,t2,t3,t4,t5,t6,a7,a6,a5,a4,a3,a2,a1,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11});

    const gpr_t vsrc = tmp.pick(); // Holds base address of src
    const gpr_t vcol = tmp.pick(); // Holds base address of col
    const gpr_t vwcol = tmp.pick(); // Holds base address of wcol
    rvjit::assembler::ld(vsrc, a0, offsetof(jit_conv_kernel_args_t, src));
    rvjit::assembler::ld(vcol, a0, offsetof(jit_conv_kernel_args_t, inter));
    const int src_sew = types::data_type_size(cfg.src_dt);
    const int col_sew = types::data_type_size(cfg.dst_dt);
    const gpr_t sew = tmp.pick();
    load_constant(sew, src_sew);
    int* w_col = new int[width_col];
    for(int i = 0; i < width_col; ++i) {
        w_col[i] = i;
    }
    intptr_t baseAddress = reinterpret_cast<intptr_t>(w_col);
    const gpr_t tmp1 = tmp.pick();

    const vr_t wcol = vout[0];
    const vr_t OFFSET = vout[1];
    const vr_t PAD = vout[2];
    const vr_t STRIDE = vout[3];
    const vr_t intermediate1 = vout[4];
    const vr_t IMCOL = vout[5];
    const vr_t WIDTHCOL = vout[6];
    const vr_t INTER = vout[7];
    const vr_t intermediate2 = vout[8];
    const vr_t colindex = vout[9];
    const vr_t WIDTH = vout[10];
    const vr_t HEIGHT = vout[11];
    const vr_t CIM = vout[12];
    const vr_t IMROW = vout[13];
    const vr_t FOUR = vout[14];
    const vr_t XERO = vout[15];
    const vr_t XERO1 = vout[16];
    const vr_t VAL = vout[17];
    const vr_t dataim = vout[18];  
    const vr_t datacol = vout[19];
    const vr_t intermediate5 = vout[20];

    const gpr_t vlen = tmp.pick();
    load_constant(vlen, cfg.vlen);
    
    const int channels_col = channels * ksize * ksize;
    // initialize channels register
    const gpr_t channel = tmp.pick();
    const gpr_t channel_end = tmp.pick();
    load_constant(channel, 0);
    load_constant(channel_end, channels_col);
    L("channels");
    //for (c = 0; c < channels_col; ++c) {

    const gpr_t ksize_reg = tmp.pick();
    load_constant(ksize_reg, ksize);
    const gpr_t w_offset = tmp.pick();
    rem(w_offset, channel, ksize_reg);
    //int w_offset = c % ksize;
    
    const gpr_t h_offset = tmp.pick();
    div(h_offset, channel, ksize_reg);
    rem(h_offset, h_offset, ksize_reg);
    //int h_offset = (c / ksize) % ksize;

    const gpr_t c_im = tmp.pick();
    div(c_im, channel, ksize_reg);
    div(c_im, c_im, ksize_reg);
    //int c_im = c / ksize / ksize;

        const gpr_t height_end = tmp.pick();
        const gpr_t height_i = tmp.pick();
        load_constant(height_i, 0);
        load_constant(height_end, height_col);
        L("heights");
        //for (h = 0; h < height_col; ++h) {

        const gpr_t im_row = tmp.pick();
        const gpr_t stride_reg = tmp.pick();
        load_constant(stride_reg, stride);
        mul(im_row, height_i, stride_reg);
        add(im_row, im_row, h_offset);
        //int im_row = h_offset + h * stride;

        const gpr_t intermediate = tmp.pick();
        mul(intermediate, c_im, height_end);
        add(intermediate, intermediate, height_i);
        //int intermediate = (c * height_col + h);

        const gpr_t pad_reg = tmp.pick();
        load_constant(pad_reg, pad);
        sub(im_row, im_row, pad_reg);
        //im_row -= pad;

        const gpr_t val = tmp.pick();
        mul(val, height_end, c_im);
        add(val, val, im_row);
        //int val = width*(im_row + height*c_im);
        
        
            const gpr_t width_i = tmp.pick();
            const gpr_t width_end = tmp.pick();
            const gpr_t index = tmp.pick();       
            load_constant(width_i, 0);
            load_constant(index, baseAddress);
            load_constant(width_end, width_col);
            L("widths");
            //for (w = 0; w < width_col; w += cfg.vlen){

                //Index calculation     
                vl(wcol, index, src_sew); // load
                addiw(index, index, src_sew*cfg.vlen); // increment
                
                vmv_sx(OFFSET, w_offset); // broadcast
                vmv_sx(PAD, pad_reg); //broadcast
                vmv_sx(STRIDE, stride_reg); //broadcast

                vmul_vv(intermediate1, STRIDE, wcol); // multiplication
                vadd_vv(IMCOL, intermediate1, OFFSET); // addition

                vmv_sx(WIDTHCOL, width_end); //broadcast

                vmv_sx(INTER, intermediate); //broadcast
                
                vmul_vv(intermediate2, INTER, WIDTHCOL); // multiplication

                vadd_vv(colindex, intermediate2, wcol); // addition

                vsub_vv(IMCOL, IMCOL, PAD); // subtract
                
                //broadcast for conditional statement
                load_constant(tmp1, width);
                vmv_sx(WIDTH, tmp1); //broadcast

                load_constant(tmp1, height);
                vmv_sx(HEIGHT, tmp1); //broadcasts

                vmv_sx(CIM, c_im); //broadcast

                vmv_sx(IMROW, im_row); //broadcast

                //Broadcast 4 for index calculation (index*4 for float 32bit)
                //int l = 4;
                vmv_sx(FOUR, sew); //broadcast
                
                int z=0;
                float z1=0.0;
                
                load_constant(tmp1, z); 
                vfsub_vv(XERO1, XERO1, XERO1); // set XERO1 to 0.0
                vmv_sx(XERO, tmp1); //broadcast
                            
                const vr_t colmask = vout[21];
                const vr_t colmask1 = vout[22];
                const vr_t colmask2 = vout[23];
                const vr_t rowmask = vout[24];
                const vr_t rowmask1 = vout[25];
                const vr_t rowmask2 = vout[26];
                const vr_t mask = vout[27];
                const vr_t mask1 = vout[28];
                const vr_t mask2 = vout[29];
                const vr_t mask3 = vout[30];
                const vr_t mask4 = vout[31];
                
                vmsgt_vx(colmask, IMCOL, XERO);
                vmslt_vx(colmask1, IMCOL, WIDTH);
                vmseq_vv(colmask2, IMCOL, XERO);
                
                vmsgt_vx(rowmask, IMROW, XERO);
                vmslt_vx(rowmask1, IMROW, HEIGHT);
                vmseq_vv(rowmask2, IMROW, XERO);
                
                vmand_mm(mask, rowmask1, colmask1);
                vmor_mm(mask1, colmask, colmask2);
                vmor_mm(mask2, rowmask, rowmask2);
                vmand_mm(mask3, mask1, mask2);
                vmand_mm(mask4, mask, mask3);
                
                //Calculate val+imcol for final index
                vmv_vx(intermediate5, val);
                // Apply mask4 to IMCOL and intermediate5
                vmul_vv(IMCOL, IMCOL, mask4);
                vmul_vv(intermediate5, intermediate5, mask4);
                vadd_vv(VAL, IMCOL, intermediate5);
                //Index multiply with 4
                vmul_vv(VAL, VAL, FOUR);
                vmul_vv(colindex, colindex, FOUR);

                vl(dataim, vsrc, src_sew);
                vmv_xs(tmp1, colindex);
                addi(vcol, vcol, tmp1);
                vs(dataim, vcol, src_sew);
                //vlox(dataim, vsrc, VAL, vmask::unmasked);
                //vsox(dataim, vcol, colindex, src_sew);
                
            // Width loop
            addi(width_i, width_i, cfg.vlen);
            blt(width_i, width_end, "widths");

        // Height loop
        addi(height_i, height_i, 1);
        blt(height_i, height_end, "heights");
    // Channel loop
    addi(channel, channel, 1);
    blt(channel, channel_end, "channels");
    
    ret();
}


/***********************3. loop interchange with manual vectorization with ALPHA!=1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling with unroll degree 24*/

void jit_convolution_kernel_t::gemm_nn_unroll16(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,
    int ii, int jj, int kk, float ALPHA, int M, int N, 
    int K,  int lda,int ldb,int ldc)
{
    /*
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
 */   
}


//6-loops with packA and PackB
void jit_convolution_kernel_t::gemm_nn_pack2(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,int M, int N, int K, float ALPHA,
        int lda, int ldb, int ldc, int BlockM, int BlockN, int BlockK, float* transposeB, float* transposeA)
    {        
    int ii,jj,kk,i,j,k;
    //int ld = __builtin_epi_vsetvlmax(__epi_e32, __epi_m1);//16;
    const gpr_t ld = tmp.pick();
    const gpr_t vlen = tmp.pick();
    load_constant(vlen, cfg.vlen);
    //vsetvli(ld, vlen, e32 | vlmul(1));
    long gvl = cfg.vlen;;

    // Registers to serve as pointers to the arguments
    const gpr_t vsrc = tmp.pick(); // A 
    const gpr_t vwei = tmp.pick(); // B 
    const gpr_t vdst = tmp.pick(); // C

    // Loading the addresses of the arguments
    rvjit::assembler::ld(vsrc, a0, offsetof(jit_conv_kernel_args_t, src));
    rvjit::assembler::ld(vwei, a0, offsetof(jit_conv_kernel_args_t, wei));
    rvjit::assembler::ld(vdst, a0, offsetof(jit_conv_kernel_args_t, dst));

    const vr_t tmp_reg = vout[0];
    const gpr_t b_index = tmp.pick();
    const gpr_t bT_index = tmp.pick();
    const gpr_t a_index = tmp.pick();
    const gpr_t aT_index = tmp.pick();
    
    const gpr_t j_loop = tmp.pick();
    const gpr_t N_reg = tmp.pick();

    const gpr_t k_loop = tmp.pick();
    const gpr_t K_reg = tmp.pick();
    auto off = asm_const(tmp, cfg.vlen * sizeof(float));
    jj = 0;
    kk = 0;
    load_constant(N_reg, N);
    if (N > 1)   load_constant(j_loop, 0);
    L("jj"); 
    //for (jj = 0; jj < N; jj+=BlockN) {
    int Nc = ((jj+BlockN>N)?(N-jj):(BlockN));
    load_constant(K_reg, K);
    if (K > 1)   load_constant(k_loop, 0);
    L("kk");
    
    //for (kk = 0; kk < K; kk+=BlockK) {
        int Kc = ((kk+BlockK > K)?(K-kk):(BlockK));
        int itr=0;
        for(int j=0;j<Nc;){
            for(int k=0;k<Kc;k++){

                //      transposeB[k*Kc+j] = B[(k+kk)*ldb+(j+jj)];
                //__epi_2xf32 tmp = __builtin_epi_vload_2xf32( &B[(k+kk)*ldb+(j+jj)], gvl);
                load_constant(b_index, ((k+kk)));//*ldb))+(j+jj)));
                add(b_index, vwei, b_index);
                vl(tmp_reg, b_index, sizeof(float));
                //svst1(pg, &transposeB[((k+(Kc*itr))*ld)+0], tmp);
                //__builtin_epi_vstore_2xf32( &transposeB[((k+(Kc*(j/ld)))*ld)+0], tmp, gvl);
                //load_constant(bT_index, (((k+(Kc*(j/ld)))*ld)+0)*sizeof(float));
                //vse32(tmp_reg, bT_index);
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
                ////load_constant(a_index, A+((i+ii)*lda+(k+kk))*sizeof(float));
                ////load_constant(aT_index, (k*Mc+i)*sizeof(float));
                ////vle32(tmp_reg, a_index);
                ////vse32(tmp_reg, aT_index);
                }
            }
            //gemm_nn_unroll16(vout, nvregs, tmp, ii,jj,kk,transposeA,transposeB, C,ALPHA, Mc,Nc, Kc, Mc,ld,ldc);
        }
    //}
    if (kk < K){
        kk+=BlockK;
        addi(k_loop, k_loop, BlockK);
        blt(k_loop, K_reg, "kk");
    }
    //}
    if (jj < N){
        jj+=BlockN;
        addi(j_loop, j_loop, BlockN);
        blt(j_loop, N_reg, "jj");
    }
    ret();
}

void jit_convolution_kernel_t::gemm_cpu(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,
        int TA, int TB, int M, int N, int K, float ALPHA, int lda, int ldb, float BETA, int ldc)
{
    
    // maybe change inputs to be like fwdd_innder_loops

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
        //size_t transposeB_ = reinterpret_cast<size_t>(transposeB);
        //size_t transposeA_ = reinterpret_cast<size_t>(transposeA);
        //gemm_nn_original(M,N,K,ALPHA,A, lda,B, ldb, C, ldc);	
        gemm_nn_pack2(vout, nvregs, tmp, M, N, K, ALPHA, lda, ldb, ldc, blockM, blockN, blockK, transposeB, transposeA);
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

    int M = oc; // Number of output channels
    int N = oh * ow; // Output spatial dimensions (flattened)
    int K = ic * kh * kw; // Dimension shared by A and B

    /// Output register block
    vr_t vout[32];
    for (int i = 0; i < nvregs; ++i)
        vout[i] = static_cast<vr_t>(i);
    /// Pool of available caller-saved general purpose registers
    register_pool_t tmp_pool({t0,t1,t2,t3,t4,t5,t6,a7,a6,a5,a4,a3,a2,a1,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11});
    im2col_cpu(vout, nvregs, tmp_pool, ic, ih, iw, kh, stride_h, l_pad);
    //gemm_cpu(vout, nvregs, tmp_pool, 0, 0, M, N, K, 1.0, K, N, 0.0, N);
    ret();
}

} // namespace gemm
} // namespace dnnl
} // namespace impl
} // namespace cpu
} // namespace riscvv