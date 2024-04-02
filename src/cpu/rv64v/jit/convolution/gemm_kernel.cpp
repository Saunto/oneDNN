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
static int label_counter = 0;
/*
void jit_convolution_kernel_t::im2col_cpu(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, 
    int channels,  int height,  int width,int ksize,  int stride, int pad)
{
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    //register_pool_t tmp_pool({t0,t1,t2,t3,t4,t5,t6,a7,a6,a5,a4,a3,a2,a1,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11});
    tmp.reset();
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
            load_constant(width_end, width_col);
            L("widths");
            //for (w = 0; w < width_col; w += cfg.vlen){
            //vsetvli(x0, vlen, vsew(sew) | vlmul(1));

                load_constant(index, baseAddress);
                //Index calculation     
                add(index, index, width_i);
                vl(wcol, index, src_sew); // load
                                
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
                //vmul_vv(IMCOL, IMCOL, mask4);
                //vmul_vv(intermediate5, intermediate5, mask4);
                vmand_mm(IMCOL, IMCOL, mask4);
                vmand_mm(intermediate5, intermediate5, mask4);
                vadd_vv(VAL, IMCOL, intermediate5);
                //Index multiply with 4
                vmul_vv(VAL, VAL, FOUR);
                vmul_vv(colindex, colindex, FOUR);

                vl(dataim, vsrc, src_sew);
                vmv_xs(tmp1, colindex);
                addi(vcol, vcol, tmp1);
                vs(dataim, vcol, src_sew);
                //vlox(dataim, vsrc, VAL, src_sew);
                //vsox(dataim, vcol, colindex, src_sew);
                
            // Width loop
            load_constant(tmp1, width_col);
            sub(tmp1, tmp1, width_i);
            addi(width_i, width_i, tmp1);
            blt(width_i, width_end, "widths");

        // Height loop
        addi(height_i, height_i, 1);
        blt(height_i, height_end, "heights");
    // Channel loop
    addi(channel, channel, 1);
    blt(channel, channel_end, "channels");

}*/
void jit_convolution_kernel_t::im2col_cpu(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, 
    int channels,  int height,  int width,int ksize,  int stride, int pad)
{
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    int channels_col = channels*ksize*ksize;
    for (int i = 0; i < nvregs; ++i)
        vout[i] = static_cast<vr_t>(i);

    //register_pool_t tmp_pool({t0,t1,t2,t3,t4,t5,t6,a7,a6,a5,a4,a3,a2,a1,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11});
    tmp.reset();
    const gpr_t vsrc = tmp.pick(); // Holds base address of src
    const gpr_t vcol = tmp.pick(); // Holds base address of col
    rvjit::assembler::ld(vsrc, a0, offsetof(jit_conv_kernel_args_t, src));
    rvjit::assembler::ld(vcol, a0, offsetof(jit_conv_kernel_args_t, inter));
    const int src_sew = types::data_type_size(cfg.src_dt);
    const int col_sew = types::data_type_size(cfg.dst_dt);
    const gpr_t sew = tmp.pick();
    load_constant(sew, src_sew);
    

    const gpr_t channel = tmp.pick();
    const gpr_t c_end = tmp.pick();
    load_constant(channel, 0);
    load_constant(c_end, channels);
    L("channels");

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

        const gpr_t ctmp = tmp.pick();
        load_constant(ctmp, height_col);
        mul(ctmp, ctmp, channel);
        const gpr_t htmp = tmp.pick();
        load_constant(htmp, height);
        mul(htmp, htmp, c_im);

        const gpr_t h = tmp.pick();
        const gpr_t h_end = tmp.pick();
        load_constant(h, 0);
        load_constant(h_end, height_col);
        L("heights");
        //for (h = 0; h < height_col; ++h) {

            const gpr_t im_row = tmp.pick();
            load_constant(im_row, stride);
            mul(im_row, im_row, h);
            add(im_row, im_row, h_offset);

            const gpr_t intermediate = tmp.pick();
            load_constant(intermediate, width_col);
            const gpr_t tmp1 = tmp.pick();
            add(tmp1, ctmp, h);
            mul(intermediate, tmp1, intermediate);
            addi(im_row, im_row, -pad);
            const gpr_t val = tmp.pick();
            add(val, htmp, im_row);
            load_constant(tmp1, width);
            mul(val, val, tmp1);

            const gpr_t w = tmp.pick();
            const gpr_t w_end = tmp.pick();
            load_constant(w, 0);
            load_constant(w_end, width_col);
            L("widths");

                const gpr_t im_col = tmp.pick();
                load_constant(im_col, stride);
                mul(im_col, im_col, w);
                add(im_col, im_col, w_offset);
                addi(im_col, im_col, -pad);
                const gpr_t col_index = tmp.pick();
                add(col_index, intermediate, w);

                //blt and bge
                mul(col_index, col_index, sew);
                add(col_index, col_index, vcol);
                
                load_constant(tmp1, 0);
                char label[16] = "result_zero";
                blt(im_row, tmp1, "result_zero");
                blt(im_col, tmp1, "result_zero");
                load_constant(tmp1, height);
                bge(im_row, tmp1, "result_zero");
                load_constant(tmp1, width);
                bge(im_col, tmp1, "result_zero");


                const gpr_t data_im = tmp.pick();
                add(data_im, im_col, val);
                mul(data_im, data_im, sew);
                add(data_im, data_im, vsrc);
                ld(tmp1, data_im, 0);
                sd(tmp1, col_index, 0);
                j("done");

                L("result_zero");
                
                load_constant(tmp1, 0);
                sd(tmp1, col_index, 0);
                
                L("done");

            addi(w, w, 1);
            blt(w, w_end, "widths");

        addi(h, h, 1);
        blt(h, h_end, "heights");


    addi(channel, channel, 1);
    blt(channel, c_end, "channels");

}

/***********************3. loop interchange with manual vectorization with ALPHA==1 double buffer ****************/
/* Manual vectorization with loop interchange + loop unrolling with unroll degree 24*/
void jit_convolution_kernel_t::gemm_nn_noalpha_unroll163loops(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,
    int M, int N, int K, float ALPHA, int A_offset, int lda, int B_offset, int ldb, int C_offset, int ldc, int unroll)
{
    tmp.reset();
    for (int i = 0; i < nvregs; ++i)
        vout[i] = static_cast<vr_t>(i);

    // Registers to serve as pointers to the arguments
    const gpr_t vsrc = tmp.pick(); // A address
    const gpr_t vwei = tmp.pick(); // B address
    const gpr_t vdst = tmp.pick(); // C address

    // Loading the addresses of the arguments
    rvjit::assembler::ld(vsrc, a0, offsetof(jit_conv_kernel_args_t, inter));
    rvjit::assembler::ld(vwei, a0, offsetof(jit_conv_kernel_args_t, wei));
    rvjit::assembler::ld(vdst, a0, offsetof(jit_conv_kernel_args_t, inter2));
    addi(vsrc, vsrc, A_offset);
    addi(vwei, vwei, B_offset);
    addi(vdst, vdst, C_offset);

    const int src_sew = types::data_type_size(cfg.src_dt);
    const gpr_t sew = tmp.pick();
    load_constant(sew, src_sew);
    const gpr_t vlen = tmp.pick();
    load_constant(vlen, cfg.vlen);
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
    const vr_t vb = vout[31]; // Using last register for B
    const vr_t vb1 = vout[16]; 
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
    if(M > 15 || (M > 7 && unroll == 8) || (M > 11 && unroll == 12) ||
    (M > 31 && unroll == 32) ||(M > 23 && unroll == 24) || (unroll == 1)){
    load_constant(j, 0);
    load_constant(N_reg, N);
    char labelj[16];
    snprintf(labelj, 16, "l%d", label_counter++);
    L(labelj);
    //for ( j = 0; j < N; ) {
        //vsetvli(x0, vlen, vsew(sew) | vlmul(1));
        switch(unroll){
            case 1: {
                load_constant(i, 0); 
                load_constant(M_reg, M);
                char labeli[16];
                snprintf(labeli, 16, "l%d", label_counter++);
                L(labeli);
                    
                    load_constant(ld_reg, ldc);
                    addi(c_index, i, 0);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);   
                    vl(vc, c_index, src_sew);

                    load_constant(k, 0);
                    load_constant(K_reg, K);
                    char labelk[16];
                    snprintf(labelk, 16, "l%d", label_counter++);
                    L(labelk);
                    //for ( k = 0; k < K; k ++) {
                        
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
                        
                        const fpr_t alpha_float = ft1;
                        flw(alpha_float, a_index, 0);
                        vfmv_sf(vaalpha, alpha_float);  
                        
                        vfmacc_vv(vc, vaalpha, vb); // sum += ALPHA*A*B                
                        
                    // k loop
                    addi(k, k, 1);
                    blt(k, K_reg, labelk);
                    
                    load_constant(ld_reg, ldc);
                    addi(c_index, i, 0);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc, c_index, src_sew);

                // i loop 
                addi(i, i, 1);
                blt(i, M_reg, labeli);
                break;
            }
            case 8: {
                load_constant(i, 0); 
                load_constant(M_reg, M-7);
                char labeli[16];
                snprintf(labeli, 16, "l%d", label_counter++);
                L(labeli);
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
                    
                    load_constant(k, 0);
                    load_constant(K_reg, K);
                    char labelk[16];
                    snprintf(labelk, 16, "l%d", label_counter++);
                    L(labelk);
                    //for ( k = 0; k < K; k ++) {
                        
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
                        
                        const fpr_t alpha_float = ft1;
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
                        
                    // k loop
                    addi(k, k, 1);
                    blt(k, K_reg, labelk);
                    
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
                    
                // i loop 
                addi(i, i, 8);
                blt(i, M_reg, labeli);
                break;
            }
            case 12: {
                load_constant(i, 0); 
                load_constant(M_reg, M-11);
                char labeli[16];
                snprintf(labeli, 16, "l%d", label_counter++);
                L(labeli);
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
                    
                    load_constant(k, 0);
                    load_constant(K_reg, K);
                    char labelk[16];
                    snprintf(labelk, 16, "l%d", label_counter++);
                    L(labelk);
                    //for ( k = 0; k < K; k ++) {
                        
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
                        
                        const fpr_t alpha_float = ft1;
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
                        
                    // k loop
                    addi(k, k, 1);
                    blt(k, K_reg, labelk);
                    
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
                    
                // i loop 
                addi(i, i, 12);
                blt(i, M_reg, labeli);
                break;
            }
            case 16: {
                load_constant(i, 0); 
                load_constant(M_reg, M-15);
                char labeli[16];
                snprintf(labeli, 16, "l%d", label_counter++);
                L(labeli);
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
                    char labelk[16];
                    snprintf(labelk, 16, "l%d", label_counter++);
                    L(labelk);
                    //for ( k = 0; k < K; k ++) {
                        
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
                        
                        const fpr_t alpha_float = ft1;
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

                        addi(b_index, k, 0);
                        mul(b_index, b_index, ld_reg);
                        add(b_index, b_index, j);
                        mul(b_index, b_index, sew);
                        add(b_index, b_index, vwei);
                        vl(vb1, b_index, src_sew);

                        addi(a_index, i, 15);
                        mul(a_index, a_index, ld_reg);
                        add(a_index, a_index, k);
                        mul(a_index, a_index, sew);
                        add(a_index, a_index, vsrc);
                        flw(alpha_float, a_index, 0);

                        vfmv_sf(vaalpha15, alpha_float);

                        vfmacc_vv(vc15, vaalpha15, vb1); // sum += ALPHA*A*B
                        vl(vb, b_index, src_sew);
                        
                        
                    // k loop
                    addi(k, k, 1);
                    blt(k, K_reg, labelk);
                    
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
                blt(i, M_reg, labeli);
                break;
            }
            case 24: {
                load_constant(i, 0); 
                load_constant(M_reg, M-23);
                char labeli[16];
                snprintf(labeli, 16, "l%d", label_counter++);
                L(labeli);
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
                    char labelk[16];
                    snprintf(labelk, 16, "l%d", label_counter++);
                    L(labelk);
                    //for ( k = 0; k < K; k ++) {
                        
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
                        
                        const fpr_t alpha_float = ft1;
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

                        addi(b_index, k, 0);
                        mul(b_index, b_index, ld_reg);
                        add(b_index, b_index, j);
                        mul(b_index, b_index, sew);
                        add(b_index, b_index, vwei);
                        vl(vb1, b_index, src_sew);

                        addi(a_index, i, 15);
                        mul(a_index, a_index, ld_reg);
                        add(a_index, a_index, k);
                        mul(a_index, a_index, sew);
                        add(a_index, a_index, vsrc);
                        flw(alpha_float, a_index, 0);

                        vfmv_sf(vaalpha15, alpha_float);

                        vfmacc_vv(vc15, vaalpha15, vb1); // sum += ALPHA*A*B
                        vl(vb, b_index, src_sew);
                        
                        
                    // k loop
                    addi(k, k, 1);
                    blt(k, K_reg, labelk);
                    
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
                    
                    load_constant(ld_reg, ldc);
                    addi(c_index, i, 16);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);   
                    vl(vc, c_index, src_sew);

                    addi(c_index, i, 17);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);   
                    vl(vc1, c_index, src_sew);

                    addi(c_index, i, 18);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc2, c_index, src_sew);

                    addi(c_index, i, 19);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc3, c_index, src_sew);

                    addi(c_index, i, 20);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc4, c_index, src_sew);

                    addi(c_index, i, 21);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc5, c_index, src_sew);

                    addi(c_index, i, 22);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc6, c_index, src_sew);

                    addi(c_index, i, 23);
                    mul(c_index, c_index, ld_reg);  
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc7, c_index, src_sew);
                    
                    load_constant(k, 0);
                    load_constant(K_reg, K);
                    char labelk1[16];
                    snprintf(labelk1, 16, "l%d", label_counter++);
                    L(labelk1);
                    //for ( k = 0; k < K; k ++) {
                        
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
                        
                    // k loop
                    addi(k, k, 1);
                    blt(k, K_reg, labelk1);
                    
                    load_constant(ld_reg, ldc);
                    addi(c_index, i, 16);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc, c_index, src_sew);

                    addi(c_index, i, 17);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc1, c_index, src_sew);

                    addi(c_index, i, 18);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc2, c_index, src_sew);

                    addi(c_index, i, 19);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc3, c_index, src_sew);

                    addi(c_index, i, 20);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc4, c_index, src_sew);

                    addi(c_index, i, 21);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc5, c_index, src_sew);

                    addi(c_index, i, 22);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc6, c_index, src_sew);

                    addi(c_index, i, 23);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc7, c_index, src_sew);
                // i loop 
                addi(i, i, 24);
                blt(i, M_reg, labeli);
                break;
            }
            case 32: {
                load_constant(i, 0); 
                load_constant(M_reg, M-31);
                char labeli[16];
                snprintf(labeli, 16, "l%d", label_counter++);
                L(labeli);
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
                    char labelk[16];
                    snprintf(labelk, 16, "l%d", label_counter++);
                    L(labelk);
                    //for ( k = 0; k < K; k ++) {
                        
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
                        
                        const fpr_t alpha_float = ft1;
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

                        addi(b_index, k, 0);
                        mul(b_index, b_index, ld_reg);
                        add(b_index, b_index, j);
                        mul(b_index, b_index, sew);
                        add(b_index, b_index, vwei);
                        vl(vb1, b_index, src_sew);

                        addi(a_index, i, 15);
                        mul(a_index, a_index, ld_reg);
                        add(a_index, a_index, k);
                        mul(a_index, a_index, sew);
                        add(a_index, a_index, vsrc);
                        flw(alpha_float, a_index, 0);

                        vfmv_sf(vaalpha15, alpha_float);

                        vfmacc_vv(vc15, vaalpha15, vb1); // sum += ALPHA*A*B
                        vl(vb, b_index, src_sew);
                        
                        
                    // k loop
                    addi(k, k, 1);
                    blt(k, K_reg, labelk);

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

                    load_constant(ld_reg, ldc);
                    addi(c_index, i, 16);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);   
                    vl(vc, c_index, src_sew);

                    addi(c_index, i, 17);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);   
                    vl(vc1, c_index, src_sew);

                    addi(c_index, i, 18);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc2, c_index, src_sew);

                    addi(c_index, i, 19);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc3, c_index, src_sew);

                    addi(c_index, i, 20);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc4, c_index, src_sew);

                    addi(c_index, i, 21);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc5, c_index, src_sew);

                    addi(c_index, i, 22);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc6, c_index, src_sew);

                    addi(c_index, i, 23);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc7, c_index, src_sew);

                    addi(c_index, i, 24);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc8, c_index, src_sew);

                    addi(c_index, i, 25);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc9, c_index, src_sew);

                    addi(c_index, i, 26);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc10, c_index, src_sew);

                    addi(c_index, i, 27);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc11, c_index, src_sew);
                
                    addi(c_index, i, 28);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc12, c_index, src_sew);

                    addi(c_index, i, 29);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc13, c_index, src_sew);

                    addi(c_index, i, 30);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc14, c_index, src_sew);

                    addi(c_index, i, 31);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);     
                    mul(c_index, c_index, sew);    
                    add(c_index, c_index, vdst);              
                    vl(vc15, c_index, src_sew);

                    load_constant(k, 0);
                    load_constant(K_reg, K);
                    char labelk1[16];
                    snprintf(labelk1, 16, "l%d", label_counter++);
                    L(labelk1);
                    //for ( k = 0; k < K; k ++) {
                        
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

                        addi(b_index, k, 0);
                        mul(b_index, b_index, ld_reg);
                        add(b_index, b_index, j);
                        mul(b_index, b_index, sew);
                        add(b_index, b_index, vwei);
                        vl(vb1, b_index, src_sew);

                        addi(a_index, i, 15);
                        mul(a_index, a_index, ld_reg);
                        add(a_index, a_index, k);
                        mul(a_index, a_index, sew);
                        add(a_index, a_index, vsrc);
                        flw(alpha_float, a_index, 0);

                        vfmv_sf(vaalpha15, alpha_float);

                        vfmacc_vv(vc15, vaalpha15, vb1); // sum += ALPHA*A*B
                        vl(vb, b_index, src_sew);
                        
                        
                    // k loop
                    addi(k, k, 1);
                    blt(k, K_reg, labelk1);
                    
                    load_constant(ld_reg, ldc);
                    addi(c_index, i, 16);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc, c_index, src_sew);

                    addi(c_index, i, 17);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc1, c_index, src_sew);

                    addi(c_index, i, 18);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc2, c_index, src_sew);

                    addi(c_index, i, 19);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc3, c_index, src_sew);

                    addi(c_index, i, 20);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc4, c_index, src_sew);

                    addi(c_index, i, 21);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc5, c_index, src_sew);

                    addi(c_index, i, 22);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc6, c_index, src_sew);

                    addi(c_index, i, 23);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc7, c_index, src_sew);

                    addi(c_index, i, 24);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc8, c_index, src_sew);

                    addi(c_index, i, 25);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc9, c_index, src_sew);

                    addi(c_index, i, 26);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc10, c_index, src_sew);

                    addi(c_index, i, 27);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc11, c_index, src_sew);

                    addi(c_index, i, 28);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc12, c_index, src_sew);

                    addi(c_index, i, 29);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc13, c_index, src_sew);

                    addi(c_index, i, 30);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc14, c_index, src_sew);

                    addi(c_index, i, 31);
                    mul(c_index, c_index, ld_reg);
                    add(c_index, c_index, j);
                    mul(c_index, c_index, sew);
                    add(c_index, c_index, vdst);
                    vs(vc15, c_index, src_sew);
                    
                // i loop 
                addi(i, i, 32);
                blt(i, M_reg, labeli);
                break;
            }
            default: {
                load_constant(i, 0); 
                load_constant(M_reg, M-15);
                char labeli[16];
                snprintf(labeli, 16, "l%d", label_counter++);
                L(labeli);
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
                    char labelk[16];
                    snprintf(labelk, 16, "l%d", label_counter++);
                    L(labelk);
                    //for ( k = 0; k < K; k ++) {
                        
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
                        
                        const fpr_t alpha_float = ft1;
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

                        addi(b_index, k, 0);
                        mul(b_index, b_index, ld_reg);
                        add(b_index, b_index, j);
                        mul(b_index, b_index, sew);
                        add(b_index, b_index, vwei);
                        vl(vb1, b_index, src_sew);

                        addi(a_index, i, 15);
                        mul(a_index, a_index, ld_reg);
                        add(a_index, a_index, k);
                        mul(a_index, a_index, sew);
                        add(a_index, a_index, vsrc);
                        flw(alpha_float, a_index, 0);

                        vfmv_sf(vaalpha15, alpha_float);

                        vfmacc_vv(vc15, vaalpha15, vb1); // sum += ALPHA*A*B
                        vl(vb, b_index, src_sew);
                        
                        
                    // k loop
                    addi(k, k, 1);
                    blt(k, K_reg, labelk);
                    
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
                blt(i, M_reg, labeli);
                break;
            }
            
        }
        
    // j loop

    //load_constant(tmp1, N);
    //sub(tmp1, tmp1, j);
    //add(j, j, tmp1);
    addi(j, j, (cfg.vlen/(src_sew*4)));
    blt(j, N_reg, labelj); 
    
    } // If M>15
    addi(i_left, i, 0);
    load_constant(j, 0);
    load_constant(N_reg, N);
    char labelj10[16];
    snprintf(labelj10, 16, "l%d", label_counter++);
    L(labelj10);

        load_constant(i, 0); 
        load_constant(M_reg, M);
        char labeli10[16];
        snprintf(labeli10, 16, "l%d", label_counter++);
        L(labeli10);
        
            load_constant(ld_reg, ldc);
            addi(c_index, i, 0);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);     
            mul(c_index, c_index, sew);    
            add(c_index, c_index, vdst);   
            vl(vc, c_index, src_sew);

            load_constant(k, 0);
            load_constant(K_reg, K);
            char labelk10[16];
            snprintf(labelk10, 16, "l%d", label_counter++);
            L(labelk10);
            //for ( k = 0; k < K; k ++) {
            
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
                
                const fpr_t alpha_float = ft1;
                flw(alpha_float, a_index, 0);
                vfmv_sf(vaalpha, alpha_float);  
                
                vfmacc_vv(vc, vaalpha, vb); // sum += ALPHA*A*B                
            
            // k loop
            addi(k, k, 1);
            blt(k, K_reg, labelk10);
        
            load_constant(ld_reg, ldc);
            addi(c_index, i, 0);
            mul(c_index, c_index, ld_reg);
            add(c_index, c_index, j);
            mul(c_index, c_index, sew);
            add(c_index, c_index, vdst);
            vs(vc, c_index, src_sew);

        // i loop 
        addi(i, i, 1);
        blt(i, M_reg, labeli10);
        
    addi(j, j, (cfg.vlen/(src_sew*4)));
    blt(j, N_reg, labelj10); 


}

void jit_convolution_kernel_t::gemm_nn_original(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, 
int M, int N, int K, float ALPHA, int A_offset, int lda, int B_offset, int ldb, int C_offset, int ldc){

    // Define block sizes, usually tuned for specific hardware
    tmp.reset();
    for (int i = 0; i < nvregs; ++i)
        vout[i] = static_cast<vr_t>(i);

    float* ALPHA_ptr = new float[1];
    ALPHA_ptr[0] = ALPHA;
    // Registers to serve as pointers to the arguments
    const gpr_t vsrc = tmp.pick(); // A address
    const gpr_t vwei = tmp.pick(); // B address
    const gpr_t vdst = tmp.pick(); // C address

    // Loading the addresses of the arguments
    rvjit::assembler::ld(vsrc, a0, offsetof(jit_conv_kernel_args_t, inter));
    rvjit::assembler::ld(vwei, a0, offsetof(jit_conv_kernel_args_t, wei));
    rvjit::assembler::ld(vdst, a0, offsetof(jit_conv_kernel_args_t, inter2));
    addi(vsrc, vsrc, A_offset);
    addi(vwei, vwei, B_offset);
    addi(vdst, vdst, C_offset);

    const int src_sew = types::data_type_size(cfg.src_dt);
    const gpr_t sew = tmp.pick();
    load_constant(sew, src_sew);
    // Loop over the blocks of the result matrix C
    const gpr_t i = tmp.pick();
    const gpr_t i_end = tmp.pick();
    load_constant(i, 0);
    load_constant(i_end, M);
    char labeli[16];
    snprintf(labeli, 16, "l%d", label_counter++);
    L(labeli);
    //for(i = 0; i < M; ++i){
        const gpr_t k = tmp.pick();
        const gpr_t k_end = tmp.pick();
        load_constant(k, 0);
        load_constant(k_end, K);
        char labelk[16];
        snprintf(labelk, 16, "l%d", label_counter++);
        L(labelk);

        //for(k = 0; k < K; ++k){
            const fpr_t alpha = ft0;
            const fpr_t A_part = ft1;
            const gpr_t float_address = tmp.pick();
            load_constant(float_address, reinterpret_cast<intptr_t>(ALPHA_ptr));
            flw(alpha, float_address, 0);
            const gpr_t a_index = tmp.pick();
            load_constant(a_index, lda);
            mul(a_index, a_index, i);
            add(a_index, a_index, k);
            add(a_index, a_index, vsrc);
            flw(A_part, a_index, 0);
            fmul_s(A_part, A_part, alpha, rm_t::rvj_rne);

            const gpr_t j = tmp.pick();
            const gpr_t j_end = tmp.pick();
            load_constant(j, 0);
            load_constant(j_end, N);
            char labelj[16];
            snprintf(labelj, 16, "l%d", label_counter++);
            L(labelj);
            //for(j = 0; j < N; ++j){
                const gpr_t b_index = tmp.pick();
                const fpr_t B_part = ft2;
                load_constant(b_index, ldb);
                mul(b_index, b_index, k);
                add(b_index, b_index, j);
                add(b_index, b_index, vwei);
                flw(B_part, b_index, 0);
                fmul_s(B_part, B_part, A_part, rm_t::rvj_rne);

                const gpr_t c_index = tmp.pick();
                load_constant(c_index, ldc);
                mul(c_index, c_index, i);
                add(c_index, c_index, j);
                add(c_index, c_index, vdst);

                const fpr_t C_part = ft3;
                flw(C_part, c_index, 0);
                fadd_s(C_part, C_part, B_part, rm_t::rvj_rne);
                fsw(c_index, C_part, 0);
                //C[i*ldc+j] += A_PART*B[k*ldb+j];

            addi(j, j, 1);
            blt(j, j_end, labelj);

        addi(k, k, 1);
        blt(k, k_end, labelk);
    
    addi(i, i, 1);
    blt(i, i_end, labeli);

}

void jit_convolution_kernel_t::blocked_gemm(convolution_schedule_t::jit_conv_kernel_args_t kargs,
    rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, int M, int N, int K, float ALPHA,
        int lda, int ldb, int ldc, int block_size_M, int block_size_N, int block_size_K)
{
 
    int current_block_size_N = block_size_N;
    int current_block_size_K = block_size_K;
    int current_block_size_M = block_size_M;
   
    for (int jj = 0; jj < N; jj += current_block_size_N) {
    current_block_size_N = std::min(block_size_N, N - jj);
    for (int kk = 0; kk < K; kk += current_block_size_K) {
        current_block_size_K = std::min(block_size_K, K - kk);
        for (int ii = 0; ii < M; ii += current_block_size_M) {
            current_block_size_M = std::min(block_size_M, M - ii);
            // Compute offsets for A, B, and C matrices
            int A_offset = ii * lda + kk;   // Offset into A matrix
            int B_offset = kk * ldb + jj;   // Offset into B matrix
            int C_offset = ii * ldc + jj;   // Offset into C matrix
            //std::cout << "A_offset: " << A_offset << " B_offset: " << B_offset << " C_offset: " << C_offset << std::endl;
            // Call the core GEMM for the current block
            gemm_nn_noalpha_unroll163loops(vout, nvregs, tmp,
                current_block_size_M, current_block_size_N, current_block_size_K, ALPHA,
                A_offset*4, lda,
                B_offset*4, ldb,
                C_offset*4, ldc,
                16
            );
            
        }
    }
}

}
void jit_convolution_kernel_t::gemm_cpu(convolution_schedule_t::jit_conv_kernel_args_t kargs, rvjit::vr_t *vout, int nvregs, register_pool_t &tmp,
        int TA, int TB, int M, int N, int K, float ALPHA, int lda, int ldb, float BETA, int ldc)
{
    int config[5] = {1,1,1,1,1};
    if(!TA && !TB)
    {
        
        if(N == 50176){
            blocked_gemm(kargs, vout, nvregs, tmp, M, N, K, ALPHA, lda, ldb, ldc, M, 
            N/config[0], K);
        }
        else if(N == 12544){
            blocked_gemm(kargs, vout, nvregs, tmp, M, N, K, ALPHA, lda, ldb, ldc, M, 
            N/config[1], K);

        }
        else if(N == 3136){
            blocked_gemm(kargs, vout, nvregs, tmp, M, N, K, ALPHA, lda, ldb, ldc, M, 
            N/config[2], K);
        }
        else if(N == 784){
            blocked_gemm(kargs, vout, nvregs, tmp, M, N, K, ALPHA, lda, ldb, ldc, M,
            N/config[3], K);
        }
        else if(N == 196){
            blocked_gemm(kargs, vout, nvregs, tmp, M, N, K, ALPHA, lda, ldb, ldc, M, 
            N/config[4], K);
        }
        else
        {
            gemm_nn_noalpha_unroll163loops(vout, nvregs, tmp, M, N, K, ALPHA, 0, lda, 0, ldb, 0, ldc, 0);
        }
        /*for(int i = 0; i < 10; i++){
            blocked_gemm(kargs, vout, nvregs, tmp, M, N, K, ALPHA, lda, ldb, ldc, M, N, K);
        }*/
        //blocked_gemm(kargs, vout, nvregs, tmp, M, N, K, ALPHA, lda, ldb, ldc, 64, 64, 64);

        //gemm_nn_noalpha_unroll163loops(vout, nvregs, tmp, M, N, K, ALPHA, 0, lda, 0, ldb, 0, ldc, 16);
        
    }/*
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);*/

}

    
void jit_convolution_kernel_t::col2im_cpu(rvjit::vr_t *vout, int nvregs, register_pool_t &tmp, 
    int channels, int height, int width, int ksize, int stride, int pad) {
    const int height_col = (height + 2 * pad - ksize) / stride + 1;
    const int width_col = (width + 2 * pad - ksize) / stride + 1;
    const int channels_col = channels * ksize * ksize;
    tmp.reset();
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

}


void jit_convolution_kernel_t::code(convolution_schedule_t::jit_conv_kernel_args_t kargs){
    const int oh = cfg.oh; // output height
    const int ow = cfg.ow; // output width
    const int ih = cfg.ih; // input height
    const int iw = cfg.iw; // input width
    const int kh = cfg.kh; // kernel height
    const int kw = cfg.kw; // kernel width
    //const int oc = cfg.oc; // output channels
    const int oc = cfg.ic;
    const int ic = cfg.ic; // input channels
    const int stride_h = cfg.stride_h; // stride height
    const int stride_w = cfg.stride_w; // stride width
    const int l_pad = cfg.l_pad; // left padding
    const int t_pad = cfg.t_pad; // top padding
    const int vlen = cfg.vlen; // vector length
    //const int nvregs = traits.erbw * traits.erbc;
    const int nvregs = 32;
    const size_t wei_sew = types::data_type_size(cfg.wei_dt);
    const size_t bia_sew = cfg.with_bias ? types::data_type_size(cfg.bias_dt) : 0;
    const size_t src_sew = types::data_type_size(cfg.src_dt);
    const size_t dst_sew = types::data_type_size(cfg.dst_dt);

    int M = oc;                   // Number of output channels
    int N = oh * ow;              // Total number of spatial positions in the output
    int K = ic * kh * kw;         // Total number of elements in a single filter
    int lda = 3 * 3 * ic;         // Elements per column in matrix A after im2col
    int ldb = 3 * 3 * ic;         // Elements per row in matrix B (filter)
    int ldc = oh * ow;            // Spatial dimensions of the output feature map

    /// Output register block
    vr_t vout[32];
    for (int i = 0; i < nvregs; ++i)
        vout[i] = static_cast<vr_t>(i);

    int size = kh * kw * ic * oh * ow;

    // Allocate memory for the intermediate data
    // Using float col[size]; does not work due to stack size limitations
    // Using float* col = new float[size]; does work since it uses the heap  
    
    /// Pool of available caller-saved general purpose registers
    register_pool_t tmp_pool({t0,t1,t2,t3,t4,t5,t6,a7,a6,a5,a4,a3,a2,a1,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11});
    im2col_cpu(vout, nvregs, tmp_pool, ic, ih, iw, kh, stride_h, l_pad);
    gemm_cpu(kargs, vout, nvregs, tmp_pool, 0, 0, M, N, K, 1.0, lda, ldb, 0.0, ldc);
    col2im_cpu(vout, nvregs, tmp_pool, oc, oh, ow, kh, stride_h, t_pad);
    ret();
}

} // namespace gemm
} // namespace dnnl
} // namespace impl
} // namespace cpu
} // namespace riscvv