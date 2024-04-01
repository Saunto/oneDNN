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
/*
void jit_convolution_kernel_t::gemm_nn_unroll16(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, int ii, int jj, int kk, float ALPHA, int M, int N, int K,  int lda,int ldb,int ldc){
    
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
  
}
*/

//6-loops with packA and PackB
/*
void jit_convolution_kernel_t::gemm_nn_pack2(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,int M, int N, int K, float ALPHA,
        int lda, int ldb, int ldc, int BlockM, int BlockN, int BlockK, float* transposeB, float* transposeA)
    {        
    //int ii,jj,kk,i,j,k;
    const gpr_t vlen = tmp.pick();
    load_constant(vlen, cfg.vlen);
    // Registers to serve as pointers to the arguments
    const gpr_t vsrc = tmp.pick(); // A address
    const gpr_t vwei = tmp.pick(); // B address
    const gpr_t vdst = tmp.pick(); // C address

    // Loading the addresses of the arguments
    rvjit::assembler::ld(vsrc, a0, offsetof(jit_conv_kernel_args_t, inter));
    rvjit::assembler::ld(vwei, a0, offsetof(jit_conv_kernel_args_t, wei));
    rvjit::assembler::ld(vdst, a0, offsetof(jit_conv_kernel_args_t, dst));
    
    auto off = asm_const(tmp, cfg.vlen * sizeof(float));

    const gpr_t jj = tmp.pick();
    const gpr_t N_reg = tmp.pick();
    load_constant(jj, 0);
    load_constant(N_reg, N);
    L("jj");
    //for (jj = 0; jj < N; jj+=BlockN) {

        const gpr_t Nc = tmp.pick();
        const gpr_t BlockN_reg = tmp.pick();
        const gpr_t tmp1 = tmp.pick();
        load_constant(BlockN_reg, BlockN);
        add(BlockN_reg, jj, BlockN_reg);

        load_constant(tmp1, BlockN);        
        blt(BlockN_reg, N_reg, "Nc");
        beq(BlockN_reg, N_reg, "Nc");
        
        sub(tmp1, N_reg, jj);

        L("Nc");
        load_constant(Nc, tmp1);
        // ((jj+BlockN>N)?(N-jj):(BlockN)));
        // ((jj+BlockN<=N)?(BlockN):(N-jj)
        
        const gpr_t kk = tmp.pick();
        const gpr_t K_reg = tmp.pick();
        load_constant(kk, 0);
        load_constant(K_reg, K);
        L("kk");
        //for (kk = 0; kk < K; kk+=BlockK) {

            const gpr_t Kc = tmp.pick();
            const gpr_t BlockK_reg = tmp.pick();
            load_constant(BlockK_reg, BlockK);
            add(BlockK_reg, kk, BlockK_reg);

            load_constant(tmp1, BlockN);        
            blt(BlockK_reg, K_reg, "Kc");
            beq(BlockK_reg, K_reg, "Kc");
            
            sub(tmp1, K_reg, kk);

            L("Kc");
            load_constant(Kc, tmp1);
            // ((kk+BlockK>K)?(K-kk):(BlockK)))
            // ((kk+BlockK<=N)?(BlockN):(N-jj)

            const gpr_t itr = tmp.pick();
            load_constant(itr, 0);
        
            const gpr_t j = tmp.pick();
            load_constant(j, 0);
            L("j");
            //for(int j=0;j<Nc; j+=gvl;){
            
                const gpr_t k = tmp.pick();
                load_constant(k, 0);
                L("k");
                //for(int k=0;k<Kc;k++){
                    intptr_t transposeBAddress = reinterpret_cast<intptr_t>(transposeB);

                    const gpr_t transposeB_index = tmp.pick();
                    const gpr_t B_index = tmp.pick();
                    div(transposeB_index, j, vlen);
                    mul(transposeB_index, transposeB_index, Kc);
                    add(transposeB_index, transposeB_index, k);
                    mul(transposeB_index, transposeB_index, vlen);
                    add(transposeB_index, transposeB_index, vwei);
                    // transposeB[((k+(Kc*(j/cfg.vlen)))*cfg.vlen)+0]

                    const gpr_t ld_temp = tmp.pick();
                    add(B_index, kk, k);
                    load_constant(ld_temp, ldb);
                    mul(B_index, B_index, ld_temp);
                    add(B_index, B_index, j);
                    add(B_index, B_index, jj);
                    add(B_index, B_index, vwei);
                    // B[(k+kk)*ldb+(j+jj)]

                    const vr_t tmp_reg = vout[0];
                    const int src_sew = types::data_type_size(cfg.src_dt);

                    vl(tmp_reg, B_index, src_sew);
                    vs(tmp_reg, transposeB_index, src_sew); 
                             
                addi(k, k, 1);
                blt(k, Kc, "k");

                addi(itr, itr, 1);
        
            addi(j, j, cfg.vlen);
            blt(j, Nc, "j");
       
        const gpr_t ii = tmp.pick();
        const gpr_t M_reg = tmp.pick();
        load_constant(ii, 0);
        load_constant(M_reg, M);
        L("ii");
        //for (ii = 0; ii < M; ii+=BlockM) {

            const gpr_t Mc = tmp.pick();
            const gpr_t BlockM_reg = tmp.pick();
            load_constant(BlockM_reg, BlockM);
            add(BlockM_reg, ii, BlockM_reg);

            load_constant(tmp1, BlockM);        
            blt(BlockM_reg, M_reg, "Mc");
            beq(BlockM_reg, M_reg, "Mc");
            
            sub(tmp1, M_reg, ii);

            L("Mc");
            load_constant(Mc, tmp1);
            // ((ii+BlockM>M)?(M-ii):(BlockM)))
            // ((ii+BlockM<=M)?(BlockM):(M-ii)
            const gpr_t itr1 = tmp.pick();
            load_constant(itr1, 0);
        
            const gpr_t i = tmp.pick();
            load_constant(i, 0);
            L("i");
            //for(int i=0;i<Mc;i++){
                load_constant(k, 0);
                L("k1");
                //for(int k=0;k<Kc;k++){
                    //transposeA[k*Mc+i] = A[(i+ii)*lda+(k+kk)];
                    intptr_t transposeAAddress = reinterpret_cast<intptr_t>(transposeA);
                    const gpr_t transposeA_index = tmp.pick();
                    mul(transposeA_index, k, Mc);
                    add(transposeA_index, transposeA_index, i);
                    // transposeA[k*Mc+i]

                    const gpr_t A_index = tmp.pick();
                    load_constant(ld_temp, lda);
                    add(A_index, ii, i);
                    mul(A_index, A_index, ld_temp);
                    add(A_index, A_index, kk);
                    add(A_index, A_index, k);
                    // A[(i+ii)*lda+(k+kk)]

                    vl(tmp_reg, A_index, src_sew);
                    vs(tmp_reg, transposeA_index, src_sew); 
                
                addi(k, k, 1);
                blt(k, Kc, "k1");
                //gemm_nn_unroll16(vout, nvregs, tmp, ii,jj,kk,transposeA,transposeB, C,ALPHA, Mc,Nc, Kc, Mc,ld,ldc);
            
            addi(i, i, 1);
            blt(i, Mc, "i");

        addi(ii, ii, BlockM);
        blt(ii, M_reg, "ii");

        // KK loop
        addi(kk, kk, BlockK);
        blt(kk, K_reg, "kk");

    // JJ loop
    addi(jj, jj, BlockN);
    blt(jj, N_reg, "jj");
    
    ret();
}*/

/***********************3. loop interchange with manual vectorization with ALPHA==1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling with unroll degree 24*/
void jit_convolution_kernel_t::gemm_nn_noalpha_unroll163loops(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,
    int M, int N, int K, float ALPHA, int lda, int ldb, int ldc)
{
    // Registers to serve as pointers to the arguments
    const gpr_t vsrc = tmp.pick(); // A address
    const gpr_t vwei = tmp.pick(); // B address
    const gpr_t vdst = tmp.pick(); // C address

    // Loading the addresses of the arguments
    rvjit::assembler::ld(vsrc, a0, offsetof(jit_conv_kernel_args_t, inter));
    rvjit::assembler::ld(vwei, a0, offsetof(jit_conv_kernel_args_t, wei));
    rvjit::assembler::ld(vdst, a0, offsetof(jit_conv_kernel_args_t, inter2));

    const int src_sew = types::data_type_size(cfg.src_dt);
    const gpr_t sew = tmp.pick();
    load_constant(sew, src_sew);
    const gpr_t vlen = tmp.pick();

    const gpr_t a_index = tmp.pick();
    const gpr_t b_index = tmp.pick();
    const gpr_t c_index = tmp.pick();

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
    const gpr_t i = tmp.pick();
    const gpr_t ld_reg = tmp.pick();
    const gpr_t j = tmp.pick();
    const gpr_t N_reg = tmp.pick();
    const gpr_t M_reg = tmp.pick();
    const gpr_t k = tmp.pick();
    const gpr_t K_reg = tmp.pick();
    const gpr_t i_left = tmp.pick();
    const gpr_t tmp1 = tmp.pick();
    load_constant(i, 0);
    if(M>15){
    load_constant(j, 0);
    load_constant(N_reg, N);
    L("j");
    //for ( j = 0; j < N; ) {
    
        load_constant(i, 0); 
        load_constant(M_reg, M-15);
        L("i");
        //for (i = 0; i < M-15; i += 16) {
            
            load_constant(ld_reg, ldc);
            addi(c_index, i, 0);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);   
            vl(vc, c_index, src_sew);

            addi(c_index, i, 1);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);   
            vl(vc1, c_index, src_sew);

            addi(c_index, i, 2);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc2, c_index, src_sew);

            addi(c_index, i, 3);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc3, c_index, src_sew);

            addi(c_index, i, 4);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc4, c_index, src_sew);

            addi(c_index, i, 5);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc5, c_index, src_sew);

            addi(c_index, i, 6);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc6, c_index, src_sew);

            addi(c_index, i, 7);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc7, c_index, src_sew);

            addi(c_index, i, 8);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc8, c_index, src_sew);

            addi(c_index, i, 9);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc9, c_index, src_sew);

            addi(c_index, i, 10);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc10, c_index, src_sew);

            addi(c_index, i, 11);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc11, c_index, src_sew);
        
        
            addi(c_index, i, 12);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc12, c_index, src_sew);

            addi(c_index, i, 13);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc13, c_index, src_sew);

            addi(c_index, i, 14);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc14, c_index, src_sew);

            addi(c_index, i, 15);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);              
            vl(vc15, c_index, src_sew);
            
            load_constant(k, 0);
            load_constant(K_reg, K);
            L("k");
            //for ( k = 0; k < K; k ++) {
                vb = vout[31]; // Using last register for B
                load_constant(ld_reg, ldb);
                addi(b_index, k, 0);
                mul(b_index, b_index, ld_reg);
                add(b_index, b_index, j);
                mul(b_index, b_index, sew);
                add(b_index, b_index, vwei);
                vl(vb, b_index, src_sew);

                load_constant(ld_reg, lda);
                addi(a_index, i, 0);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);

                const fpr_t alpha_float = rvj_ft0;
                flw(alpha_float, a_index, 0);
                vfmv_sf(vaalpha, alpha_float);  
                
                vfmacc_vv(vc, vaalpha, vb); // sum += ALPHA*A*B

                addi(a_index, i, 1);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);
                vfmv_sf(vaalpha1, alpha_float);

                vfmacc_vv(vc1, vaalpha1, vb); // sum += ALPHA*A*B

                addi(a_index, i, 2);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);
                vfmv_sf(vaalpha2, alpha_float);

                vfmacc_vv(vc2, vaalpha2, vb); // sum += ALPHA*A*B

                addi(a_index, i, 3);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);
                vfmv_sf(vaalpha3, alpha_float);
                
                vfmacc_vv(vc3, vaalpha3, vb); // sum += ALPHA*A*B

                addi(a_index, i, 4); 
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);
                vfmv_sf(vaalpha4, alpha_float);

                vfmacc_vv(vc4, vaalpha4, vb); // sum += ALPHA*A*B

                addi(a_index, i, 5);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);
                vfmv_sf(vaalpha5, alpha_float);

                vfmacc_vv(vc5, vaalpha5, vb); // sum += ALPHA*A*B

                addi(a_index, i, 6);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);
                vfmv_sf(vaalpha6, alpha_float);

                vfmacc_vv(vc6, vaalpha6, vb); // sum += ALPHA*A*B

                addi(a_index, i, 7);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);

                vfmv_sf(vaalpha7, alpha_float);

                vfmacc_vv(vc7, vaalpha7, vb); // sum += ALPHA*A*B

                addi(a_index, i, 8);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);

                vfmv_sf(vaalpha8, alpha_float);

                vfmacc_vv(vc8, vaalpha8, vb); // sum += ALPHA*A*B

                addi(a_index, i, 9);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);

                vfmv_sf(vaalpha9, alpha_float);

                vfmacc_vv(vc9, vaalpha9, vb); // sum += ALPHA*A*B

                addi(a_index, i, 10);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);

                vfmv_sf(vaalpha10, alpha_float);

                vfmacc_vv(vc10, vaalpha10, vb); // sum += ALPHA*A*B

                addi(a_index, i, 11);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);

                vfmv_sf(vaalpha11, alpha_float);

                vfmacc_vv(vc11, vaalpha11, vb); // sum += ALPHA*A*B

                addi(a_index, i, 12);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);

                vfmv_sf(vaalpha12, alpha_float);

                vfmacc_vv(vc12, vaalpha12, vb); // sum += ALPHA*A*B

                addi(a_index, i, 13);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);

                vfmv_sf(vaalpha13, alpha_float);

                vfmacc_vv(vc13, vaalpha13, vb); // sum += ALPHA*A*B

                addi(a_index, i, 14);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);

                vfmv_sf(vaalpha14, alpha_float);

                vfmacc_vv(vc14, vaalpha14, vb); // sum += ALPHA*A*B
                vb = vout[16];
                addi(b_index, k, 0);
                mul(b_index, b_index, ld_reg);
                add(b_index, b_index, j);
                mul(b_index, b_index, sew);
                add(b_index, b_index, vwei);
                vl(vb, b_index, src_sew);

                addi(a_index, i, 15);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);

                vfmv_sf(vaalpha15, alpha_float);

                vfmacc_vv(vc15, vaalpha15, vb); // sum += ALPHA*A*B
                
            // k loop
            addi(k, k, 1);
            blt(k, K_reg, "k");
            
            load_constant(ld_reg, ldc);
            addi(c_index, i, 0);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc, c_index, src_sew);

            addi(c_index, i, 1);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc1, c_index, src_sew);

            addi(c_index, i, 2);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc2, c_index, src_sew);

            addi(c_index, i, 3);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc3, c_index, src_sew);

            addi(c_index, i, 4);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc4, c_index, src_sew);

            addi(c_index, i, 5);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc5, c_index, src_sew);

            addi(c_index, i, 6);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc6, c_index, src_sew);

            addi(c_index, i, 7);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc7, c_index, src_sew);

            addi(c_index, i, 8);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc8, c_index, src_sew);

            addi(c_index, i, 9);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc9, c_index, src_sew);

            addi(c_index, i, 10);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc10, c_index, src_sew);

            addi(c_index, i, 11);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc11, c_index, src_sew);

            addi(c_index, i, 12);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc12, c_index, src_sew);

            addi(c_index, i, 13);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc13, c_index, src_sew);

            addi(c_index, i, 14);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc14, c_index, src_sew);

            addi(c_index, i, 15);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc15, c_index, src_sew);
            
        // i loop 
        addi(i, i, 16);
        blt(i, M_reg, "i");
    // j loop
    addi(j, j, cfg.vlen);
    blt(j, N_reg, "j"); 
    
    } // If M>15
    
   
    addi(i_left, i, 0);
    load_constant(j, 0);
    L("j1");
    //for (int j = 0; j < N; j += gvl;) {
        
        addi(i, i_left, 0);
        load_constant(M_reg, M);
        L("i1");
        //for (i=i_left; i < M; i += 4) { // change according to unroll degree
            load_constant(ld_reg, ldc);
            addi(c_index, i, 0);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vl(vc, c_index, src_sew);

            addi(tmp1, i, 1);
            bge(tmp1, M_reg, "vc2");
            addi(c_index, i, 1);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vl(vc1, c_index, src_sew);

            L("vc2");
            addi(tmp1, i, 2);
            bge(tmp1, M_reg, "vc3");
            addi(c_index, i, 2);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vl(vc2, c_index, src_sew);

            L("vc3");
            addi(tmp1, i, 3);
            bge(tmp1, M_reg, "skip");
            addi(c_index, i, 3);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vl(vc3, c_index, src_sew);

            L("skip");

            load_constant(k, 0);
            load_constant(K_reg, K);
            L("k1");
            //for (int k = 0; k < K; k ++) {
                load_constant(ld_reg, lda);
                addi(a_index, i, 0);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                const fpr_t alpha_float = rvj_ft0;
                flw(alpha_float, a_index, 0);
                vfmv_sf(vaalpha, alpha_float);

                addi(tmp1, i, 1);
                bge(tmp1, M_reg, "va2");
                addi(a_index, i, 1);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);  
                vfmv_sf(vaalpha1, alpha_float);

                L("va2");
                addi(tmp1, i, 2);
                bge(tmp1, M_reg, "va3");
                addi(a_index, i, 2);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);
                vfmv_sf(vaalpha2, alpha_float);

                L("va3");
                addi(tmp1, i, 3);
                bge(tmp1, M_reg, "skip1");
                addi(a_index, i, 3);
                mul(a_index, a_index, ld_reg);
                add(a_index, a_index, k);
                mul(a_index, a_index, sew);
                add(a_index, a_index, vsrc);
                flw(alpha_float, a_index, 0);
                vfmv_sf(vaalpha3, alpha_float);

                L("skip1");

                load_constant(ld_reg, ldb);
                addi(b_index, k, 0);
                mul(b_index, b_index, ld_reg);
                add(b_index, b_index, j);
                mul(b_index, b_index, sew);
                add(b_index, b_index, vwei);
                vl(vb, b_index, src_sew);    

                vfmacc_vv(vc, vaalpha, vb); // sum += ALPHA*A*B

                addi(tmp1, i, 1);
                bge(tmp1, M_reg, "vc21");
                vfmacc_vv(vc1, vaalpha1, vb); // sum += ALPHA*A*B

                L("vc21");
                addi(tmp1, i, 2);
                bge(tmp1, M_reg, "vc31");
                vfmacc_vv(vc2, vaalpha2, vb); // sum += ALPHA*A*B
                
                L("vc31");
                addi(tmp1, i, 3);
                bge(tmp1, M_reg, "skip2");
                vfmacc_vv(vc3, vaalpha3, vb); // sum += ALPHA*A*B
                
                L("skip2");


            addi(k, k, 1);
            blt(k, K_reg, "k1");

            load_constant(ld_reg, ldc);
            addi(c_index, i, 0);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc, c_index, src_sew);

            //if (i+1 < M)      {__builtin_epi_vstore_2xf32(&C[(i+1)*ldc+j], vc1, gvl);}
            //if (i+2 < M)      {__builtin_epi_vstore_2xf32(&C[(i+2)*ldc+j], vc2, gvl);}
            //if (i+3 < M)      {__builtin_epi_vstore_2xf32(&C[(i+3)*ldc+j], vc3, gvl);}
            L("vc12");
            addi(tmp1, i, 1);
            bge(tmp1, M_reg, "vc22");
            addi(c_index, i, 1);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc1, c_index, src_sew);

            L("vc22");
            addi(tmp1, i, 2);
            bge(tmp1, M_reg, "vc32");
            addi(c_index, i, 2);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc2, c_index, src_sew);

            L("vc32");
            addi(tmp1, i, 3);
            bge(tmp1, M_reg, "skip3");
            addi(c_index, i, 3);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc3, c_index, src_sew);
                        
            L("skip3");
        // i1 loop
        addi(i, i, 4); // change according to unroll degree
        blt(i, M_reg, "i1");

    // j1 loop
    addi(j, j, cfg.vlen);
    blt(j, N_reg, "j1");
    ret();
}

void jit_convolution_kernel_t::gemm_cpu(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,
        int TA, int TB, int M, int N, int K, float ALPHA, int lda, int ldb, float BETA, int ldc)
{
    if(!TA && !TB)
    {
    /*
	    // enable below for the 6-loops packed implementations 
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
    */
    /*** 3-loop implementation */
	    gemm_nn_noalpha_unroll163loops(vout, nvregs, tmp, M, N, K, ALPHA,lda, ldb,ldc);
    }/*
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);*/
    
    ret();
}

    
void jit_convolution_kernel_t::col2im_cpu(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, 
    int channels, int height, int width, int ksize, int stride, int pad) {
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;

    const gpr_t vsrc = tmp.pick(); // Holds base address of src/col
    const gpr_t vdst = tmp.pick(); // Holds base address of dst/im
    rvjit::assembler::ld(vsrc, a0, offsetof(jit_conv_kernel_args_t, inter2));
    rvjit::assembler::ld(vdst, a0, offsetof(jit_conv_kernel_args_t, dst));
    const int src_sew = types::data_type_size(cfg.src_dt);
    const gpr_t sew = tmp.pick();
    load_constant(sew, src_sew);

    const gpr_t channel_reg = tmp.pick();
    const gpr_t channels_end = tmp.pick();
    load_constant(channel_reg, 0);
    load_constant(channels_end, channels_col);
    L("c");
    //for (int c = 0; c < channels_col; ++c) {
        const gpr_t ksize_reg = tmp.pick();
        load_constant(ksize_reg, ksize);
        const gpr_t w_offset = tmp.pick();
        rem(w_offset, channel_reg, ksize_reg);
        //int w_offset = c % ksize;
        
        const gpr_t h_offset = tmp.pick();
        div(h_offset, channel_reg, ksize_reg);
        rem(h_offset, h_offset, ksize_reg);
        //int h_offset = (c / ksize) % ksize;

        const gpr_t c_im = tmp.pick();
        div(c_im, channel_reg, ksize_reg);
        div(c_im, c_im, ksize_reg);
        //int c_im = c / ksize / ksize;

        const gpr_t height_reg = tmp.pick();
        const gpr_t height_end = tmp.pick();
        load_constant(height_reg, 0);
        load_constant(height_end, height_col);
        L("h");
        //for (int h = 0; h < height_col; ++h) {
            const gpr_t width_reg = tmp.pick();
            const gpr_t width_end = tmp.pick();
            load_constant(width_reg, 0);
            load_constant(width_end, width_col);
            L("w");
            //for (int w = 0; w < width_col; ++w) {
                const gpr_t im_row = tmp.pick();
                const gpr_t im_col = tmp.pick();
                addi(im_row, height_reg, 0);
                const gpr_t stride_reg = tmp.pick();
                load_constant(stride_reg, stride);
                mul(im_row, im_row, stride_reg);
                const gpr_t pad_reg = tmp.pick();
                load_constant(pad_reg, pad);
                sub(im_row, im_row, pad_reg);
                add(im_row, im_row, h_offset);
                //int im_row = h_offset + h * stride - pad;
                addi(im_col, width_reg, 0);
                mul(im_col, im_col, stride_reg);
                sub(im_col, im_col, pad_reg);
                add(im_col, im_col, w_offset);
                //int im_col = w_offset + w * stride - pad;
                const gpr_t zero = tmp.pick();
                load_constant(zero, 0);
                blt(im_row, zero, "continue");
                blt(im_col, zero, "continue");
                bge(im_row, height_reg, "continue");
                bge(im_col, width_reg, "continue");

                const gpr_t col_index = tmp.pick();
                load_constant(col_index, height_col);
                mul(col_index, col_index, channel_reg);
                add(col_index, col_index, height_reg);
                mul(col_index, col_index, width_end);
                add(col_index, col_index, width_reg);
                mul(col_index, col_index, sew);
                add(col_index, col_index, vsrc);

                const gpr_t data_col = tmp.pick();
                ld(data_col, col_index, 0);

                const gpr_t im_index = tmp.pick();
                load_constant(im_index, height_end);
                mul(im_index, im_index, c_im);
                add(im_index, im_index, im_row);
                mul(im_index, im_index, width_end);
                add(im_index, im_index, im_col);
                mul(im_index, im_index, sew);
                add(im_index, im_index, vdst);
                sd(data_col, im_index, 0);

                L("continue");
            // w loop
            addi(width_reg, width_reg, 1);
            blt(width_reg, width_end, "w");
        // h loop
        addi(height_reg, height_reg, 1);
        blt(height_reg, height_end, "h");
    // c loop
    addi(channel_reg, channel_reg, 1);
    blt(channel_reg, channels_end, "c");

    ret();
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
    gemm_cpu(vout, nvregs, tmp_pool, 0, 0, M, N, K, 1.0, K, N, 0.0, N);
    col2im_cpu(vout, nvregs, tmp_pool, oc, oh, ow, kh, stride_h, t_pad);
    ret();
}

} // namespace gemm
} // namespace dnnl
} // namespace impl
} // namespace cpu
} // namespace riscvv