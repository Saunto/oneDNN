#include <stdlib.h>
#include <string.h>
#include "instruction_types.h"

#define REGX rvj_gpr
#define REGF rvj_fpr
#define REGV rvj_vr

static const int nregs = 32;
char* gpr[] = {
    "x0", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0",
    "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4", "s5",
    "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"
};

char *fpr[] = {
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1",
    "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "fs2", "fs3",
    "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11", "ft8", "ft9",
    "ft10", "ft11"
};

char *vr[] = {
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
    "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
    "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
};

rvj_status_t rvj_get_gpr_id(const char * const name, rvj_gpr *out) {
    if (!name)
        return rvj_invalid_arguments;
    const int n = strlen(name);
    if (n < 2 || n > 4)
        return rvj_invalid_arguments;
    for (int i = 0; i < nregs; ++i) {
        if (strcmp(name, gpr[i]) == 0) {
            *out = i;
            return rvj_success;
        }
    }
    return rvj_invalid_arguments;
}

rvj_status_t rvj_get_fpr_id(const char * const name, rvj_fpr *out) {
    if (!name)
        return rvj_invalid_arguments;
    const int n = strlen(name);
    if (n < 2 || n > 4)
        return rvj_invalid_arguments;
    for (int i = 0; i < nregs; ++i) {
        if (strcmp(name, fpr[i]) == 0) {
            *out = i;
            return rvj_success;
        }
    }
    return rvj_invalid_arguments;
}

rvj_status_t rvj_get_vr_id(const char * const name, rvj_vr *out) {
    if (!name)
        return rvj_invalid_arguments;
    const int n = strlen(name);
    if (n < 2 || n > 4)
        return rvj_invalid_arguments;
    for (int i = 0; i < nregs; ++i) {
        if (strcmp(name, vr[i]) == 0) {
            *out = i;
            return rvj_success;
        }
    }
    return rvj_invalid_arguments;
}

int rvj_is_type_b(rvj_instr i) {
    return (i & 0x7F) == 0x63;
}

int rvj_is_jal(rvj_instr i) {
    return (i & 0x7F) == 0x67;
}

/******************************************************************************
** Mask for immediate values in different RISC-V instruction formats
******************************************************************************/

// TODO: Error detection for immediates

rvj_instr imm_type_i(int imm) {
    return (imm & 0xFFF) << 20;
}

rvj_instr imm_type_s(int imm) {
    rvj_instr imm_11_5 = (imm & 0xFE0) << 20;
    rvj_instr imm_4_0  = (imm & 0x1F) << 7;
    return imm_4_0 | imm_11_5;
}

rvj_instr imm_type_b(int imm) {
    rvj_instr imm_4_1 = (imm & 0xF) << 8;
    rvj_instr imm_10_5 = (imm & 0x3F0) << 21;
    rvj_instr imm_11 = (imm & 0x400) >> 3;
    rvj_instr imm_12 = (imm & 0x800) << 20;
    return imm_4_1 | imm_10_5 | imm_11 | imm_12;
}

rvj_instr imm_type_u(int imm) {
    return (imm & 0xFFFFF) << 12;
}

rvj_instr imm_type_j(int imm) {
    rvj_instr imm_10_1 = (imm & 0x3FF) << 21;
    rvj_instr imm_11 = (imm & 0x400 ) << 10;
    rvj_instr imm_19_12 = (imm & 0x7F800) << 1;
    rvj_instr imm_20 = (imm & 0x80000) << 12;
    return imm_10_1 | imm_11 | imm_19_12 | imm_20;
}

/******************************************************************************
** Functions to populate bitfields for each RISC-V instruction format
******************************************************************************/

/// @brief Creates an R-type instruction from its bitfields
rvj_instr opR(int rd, int rs1, int rs2, int op, int f3, int f7) {
    return (op & 0x7F) | ((rd & 0x1F) << 7) | ((f3 & 0x7) << 12)
        | ((rs1 & 0x1F) << 15) | ((rs2 & 0x1F) << 20) | ((f7 & 0x7F) << 25);
}
/// @brief Creates an R4-type instruction from its bitfields
rvj_instr opR4(int rd, int rs1, int rs2, int rs3, int op,
               int f3, int f2) {
    return (op & 0x7F) | ((rd & 0x1F) << 7) | ((f3 & 0x7) << 12)
        | ((rs1 & 0x1F) << 15) | ((rs2 & 0x1F) << 20) | (f2 << 25)
        | (rs3 << 27);
}

/// @brief Creates an I-type instruction from its bitfields
rvj_instr opI(int rd, int rs1, int imm, int op, int f3) {
    return (op & 0x7F) | imm_type_i(imm) | ((rd & 0x1F) << 7)
        | ((f3 & 0x7) << 12) | ((rs1 & 0x1F) << 15);
}

/// @brief Creates a S-type instruction from its bitfields
rvj_instr opS(int rs1, int rs2, int imm, int op, int f3) {
    return (op & 0x7F) | imm_type_s(imm) | ((f3 & 0x7) << 12)
        | ((rs1 & 0x1F) << 15) | ((rs2 & 0x1F) << 20);
}

/// @brief Creates a B-type instruction from its bitfields
rvj_instr opB(int rs1, int rs2, int imm, int f3) {
    return 0x63 | imm_type_b(imm) | ((f3 & 0x7) << 12)
        | ((rs1 & 0x1F) << 15) | ((rs2 & 0x1F) << 20);
}

/// @brief Creates an U-type instruction from its bitfields
rvj_instr opU(int rd, int imm, int op) {
    return (op & 0x7F) | ((rd & 0x1F) << 7) | imm_type_u(imm);
}

/// @brief Creates a J-type instruction from its bitfields
rvj_instr opJ(int rd, int imm, int op) {
    return (op & 0x7F) | imm_type_j(imm) | ((rd & 0x1F) << 7);
}

/// @brief Creates an FP-type instruction from its bitfields
rvj_instr opFP(int rd, int rs1, int rs2, int f5, int fmt, rvj_rm rm) {
    return 0x53 | ((rd & 0x1F) << 7) | ((rm & 0x7) << 12)
        | ((rs1 & 0x1F) << 15) | ((rs2 & 0x1F) << 20) | ((fmt & 0x3) << 25)
        | (f5 << 27);
}

/// @brief Creates a V-type instruction from its bitfields
rvj_instr opV(int vd, int vs1, int vs2, int f6,int width, int vm) {
    return 0x57 | ((vd & 0x1F) << 7) | ((width & 0x7) << 12)
        | ((vs1 & 0x1F) << 15) | ((vs2 & 0x1F) << 20) | ((vm & 1) << 25)
        | ((f6 & 0x3F) << 26);
}

/// @brief Creates a int vector-vector V-type instruction from its bitfields
rvj_instr opIVV(int vd, int vs1, int vs2, int f6, int vm) {
    return opV(vd, vs1, vs2, f6, 0, vm);
}

/// @brief Creates a float vector-vector V-type instruction from its bitfields
rvj_instr opFVV(int vd, int vs1, int vs2, int f6, int vm) {
    return opV(vd, vs1, vs2, f6, 1, vm);
}

/// @brief Creates a M vector-vector V-type instruction from its bitfields
rvj_instr opMVV(int vd, int vs1, int vs2, int f6, int vm) {
    return opV(vd, vs1, vs2, f6, 2, vm);
}

/// @brief Creates a vector-immediate V-type instruction from its bitfields
rvj_instr opIVI(int vd, int simm5, int vs2, int f6, int vm) {
    return opV(vd, (simm5 & 0x1F), vs2, f6, 3, vm);
}

/// @brief Creates a vector-scalar (GPR) V-type instruction from its bitfields
rvj_instr opIVX(int vd, int rs1, int vs2, int f6, int vm) {
    return opV(vd, rs1, vs2, f6, 4, vm);
}

/// @brief Creates a vector-scalar (FP) V-type instruction from its bitfields
rvj_instr opFVF(int vd, int rs1, int vs2, int f6, int vm) {
    return opV(vd, rs1, vs2, f6, 5, vm);
}

/// @brief Creates a vector-scalar (GPR) V-type instruction from its bitfields
rvj_instr opMVX(int vd, int rs1, int vs2, int f6, int vm) {
    return opV(vd, rs1, vs2, f6, 6, vm);
}

/******************************************************************************
** RV32-I Base Instruction Set
******************************************************************************/

// TODO: Error detection for immediates, offsets, and other bit fields

rvj_instr rvj_lui(REGX rd, int imm) {
    return opU(rd, imm, 0x37);
}

rvj_instr rvj_auipc(REGX rd, int imm) {
    return opU(rd, imm, 0x17);
}

rvj_instr rvj_jal(REGX rd, int offset) {
    return opJ(rd, offset, 0x6F);
}

rvj_instr rvj_jalr(REGX rd, REGX rs1, int offset) {
    return opI(rd, rs1, offset, 0x67, 0);
}

rvj_instr rvj_beq(REGX rs1, REGX rs2, int offset) {
    return opB(rs1, rs2, offset, 0);
}

rvj_instr rvj_bne(REGX rs1, REGX rs2, int offset) {
    return opB(rs1, rs2, offset, 1);
}

rvj_instr rvj_blt(REGX rs1, REGX rs2, int offset) {
    return opB(rs1, rs2, offset, 4);
}

rvj_instr rvj_bge(REGX rs1, REGX rs2, int offset) {
    return opB(rs1, rs2, offset, 5);
}

rvj_instr rvj_bltu(REGX rs1, REGX rs2, int offset) {
    return opB(rs1, rs2, offset, 6);
}

rvj_instr rvj_bgeu(REGX rs1, REGX rs2, int offset) {
    return opB(rs1, rs2, offset, 7);
}

rvj_instr rvj_lb(REGX rd, REGX rs1, int offset) {
    return opI(rd, rs1, offset, 3, 0);
}

rvj_instr rvj_lh(REGX rd, REGX rs1, int offset) {
    return opI(rd, rs1, offset, 3, 1);
}

rvj_instr rvj_lw(REGX rd, REGX rs1, int offset) {
    return opI(rd, rs1, offset, 3, 2);
}

rvj_instr rvj_lbu(REGX rd, REGX rs1, int offset) {
    return opI(rd, rs1, offset, 3, 4);
}

rvj_instr rvj_lhu(REGX rd, REGX rs1, int offset) {
    return opI(rd, rs1, offset, 3, 5);
}

rvj_instr rvj_sb(REGX src, REGX base, int offset) {
    return opS(base, src, offset, 0x23, 0);
}

rvj_instr rvj_sh(REGX src, REGX base, int offset) {
    return opS(base, src, offset, 0x23, 1);
}

rvj_instr rvj_sw(REGX src, REGX base, int offset) {
    return opS(base, src, offset, 0x23, 2);
}

rvj_instr rvj_addi(REGX rd, REGX rs1, int imm) {
    return opI(rd, rs1, imm, 0x13, 0);
}

rvj_instr rvj_slti(REGX rd, REGX rs1, int imm) {
    return opI(rd, rs1, imm, 0x13, 2);
}

rvj_instr rvj_sltiu(REGX rd, REGX rs1, int imm) {
    return opI(rd, rs1, imm, 0x13, 3);
}

rvj_instr rvj_xori(REGX rd, REGX rs1, int imm) {
    return opI(rd, rs1, imm, 0x13, 4);
}

rvj_instr rvj_ori(REGX rd, REGX rs1, int imm) {
    return opI(rd, rs1, imm, 0x13, 6);
}

rvj_instr rvj_andi(REGX rd, REGX rs1, int imm) {
    return opI(rd, rs1, imm, 0x13, 7);
}

rvj_instr rvj_slli(REGX rd, REGX rs1, int shamt) {
    return opI(rd, rs1, shamt, 0x13, 1);
}

rvj_instr rvj_srli(REGX rd, REGX rs1, int shamt) {
    return opI(rd, rs1, shamt, 0x13, 5);
}

rvj_instr rvj_srai(REGX rd, REGX rs1, int shamt) {
    return opI(rd, rs1, 0x400+shamt, 0x13, 5);
}

rvj_instr rvj_add(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 0, 0);
}

rvj_instr rvj_sub(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 0, 0x20);
}

rvj_instr rvj_sll(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 1, 0);
}

rvj_instr rvj_slt(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 2, 0);
}

rvj_instr rvj_sltu(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 3, 0);
}

rvj_instr rvj_xor_(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 4, 0);
}

rvj_instr rvj_srl(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 5, 0);
}

rvj_instr rvj_sra(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 5, 0x20);
}

rvj_instr rvj_or_(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 6, 0);
}

rvj_instr rvj_and_(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 7, 0);
}

rvj_instr rvj_fence(REGX rd, REGX rs1, int imm) {
    return opI(rd, rs1, imm, 0xF, 0);
}

rvj_instr rvj_ecall() {
    return 0x73;
}

rvj_instr rvj_ebreak() {
    return 0x100073;
}

/******************************************************************************
** RV32-I Pseudo Instructions
******************************************************************************/

rvj_instr rvj_nop() {
    return rvj_addi(rvj_x0, rvj_x0, rvj_x0);
}

rvj_instr rvj_ret() {
    return rvj_jalr(rvj_x0, rvj_ra, rvj_x0);
}

rvj_instr rvj_mv(REGX rd, REGX rs1) {
    return rvj_addi(rd, rs1, rvj_x0);
}

rvj_instr rvj_li(REGX rd, const int imm) {
    return rvj_addi(rd, rvj_x0, imm);
}

rvj_instr rvj_j(const int imm) {
    return rvj_jal(rvj_x0, imm);
}

rvj_instr rvj_beqz(REGX rd, int offset) {
    return rvj_beq(rd, rvj_x0, offset);
}

rvj_instr rvj_bnez(REGX rd, int offset) {
    return rvj_bne(rd, rvj_x0, offset);
}

rvj_instr rvj_sext_w(REGX rd, REGX rs1) {
    return rvj_addiw(rd, rs1, rvj_x0);
}

/******************************************************************************
** RV64-I Standard Extension
******************************************************************************/

rvj_instr rvj_lwu(REGX rd, REGX rs1, int offset) {
    return opI(rd, rs1, offset, 3, 6);
}

rvj_instr rvj_ld(REGX rd, REGX rs1, int offset) {
    return opI(rd, rs1, offset, 3, 3);
}

rvj_instr rvj_sd(REGX src, REGX base, int offset) {
    return opS(base, src, offset, 0x23, 3);
}

rvj_instr rvj_addiw(REGX rd, REGX rs1, int imm) {
    return opI(rd, rs1, imm, 0x1B, 0);
}

rvj_instr rvj_slliw(REGX rd, REGX rs1, int shamt) {
    return opI(rd, rs1, shamt, 0x1B, 1);
}

rvj_instr rvj_srliw(REGX rd, REGX rs1, int shamt) {
    return opI(rd, rs1, shamt, 0x1B, 5);
}

rvj_instr rvj_sraiw(REGX rd, REGX rs1, int shamt) {
    return opI(rd, rs1, 0x400+shamt, 0x1B, 5);
}

rvj_instr rvj_addw(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x3B, 0, 0);
}

rvj_instr rvj_subw(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x3B, 0, 0x20);
}

rvj_instr rvj_sllw(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x3B, 1, 0);
}

rvj_instr rvj_srlw(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x3B, 5, 0);
}

rvj_instr rvj_sraw(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x3B, 5, 0x20);
}

/******************************************************************************
** RV32/RV64 Zifencei Standard Extension
******************************************************************************/

rvj_instr rvj_fence_i(REGX rd, REGX rs1, int imm) {
    return opI(rd, rs1, imm, 0xF, 1);
}

/******************************************************************************
** RV32/RV64 Zicsr Standard Extension
******************************************************************************/

rvj_instr rvj_csrrw(REGX rd, REGX rs1, unsigned int csr) {
    return opI(rd, rs1, csr, 0x73, 1);
}

rvj_instr rvj_csrrs(REGX rd, REGX rs1, unsigned int csr) {
    return opI(rd, rs1, csr, 0x73, 2);
}

rvj_instr rvj_csrrc(REGX rd, REGX rs1, unsigned int csr) {
    return opI(rd, rs1, csr, 0x73, 3);
}

rvj_instr rvj_csrrwi(REGX rd, unsigned int uimm, unsigned int csr) {
    return opI(rd, uimm, csr, 0x73, 5);
}

rvj_instr rvj_csrrsi(REGX rd, unsigned int uimm, unsigned int csr) {
    return opI(rd, uimm, csr, 0x73, 6);
}

rvj_instr rvj_csrrci(REGX rd, unsigned int uimm, unsigned int csr) {
    return opI(rd, uimm, csr, 0x73, 7);
}

/******************************************************************************
** RV32-M Zicsr Standard Extension
******************************************************************************/

rvj_instr rvj_mul(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 0, 1);
}

rvj_instr rvj_mulh(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 1, 1);
}

rvj_instr rvj_mulhsu(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 2, 1);
}

rvj_instr rvj_mulhu(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 3, 1);
}

rvj_instr rvj_div(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 4, 1);
}

rvj_instr rvj_divu(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 5, 1);
}

rvj_instr rvj_rem(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 6, 1);
}

rvj_instr rvj_remu(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x33, 7, 1);
}

/******************************************************************************
** RV64-M Zicsr Standard Extension
******************************************************************************/

rvj_instr rvj_mulw(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x3B, 0, 1);
}

rvj_instr rvj_divw(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x3B, 1, 1);
}

rvj_instr rvj_divuw(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x3B, 5, 1);
}

rvj_instr rvj_remw(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x3B, 6, 1);
}

rvj_instr rvj_remuw(REGX rd, REGX rs1, REGX rs2) {
    return opR(rd, rs1, rs2, 0x3B, 7, 1);
}

/******************************************************************************
** RV32-A Standard Extension
******************************************************************************/

/******************************************************************************
** RV64-A Standard Extension
******************************************************************************/

/******************************************************************************
** RV32-F Standard Extension
******************************************************************************/

rvj_instr rvj_flw(REGF rd, REGX rs1, int offset) {
    return opI(rd, rs1, offset, 7, 2);
}

rvj_instr rvj_fsw(REGX rd, REGF rs1, int offset) {
    return opS(rd, rs1, offset, 0x27, 2);
}

rvj_instr rvj_fmadd_s(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x43, rm, 0);
}

rvj_instr rvj_fmsub_s(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x47, rm, 0);
}

rvj_instr rvj_fnmsub_s(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x4B, rm, 0);
}

rvj_instr rvj_fnmadd_s(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x4F, rm, 0);
}

rvj_instr rvj_fadd_s(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 0, 0, rm);
}

rvj_instr rvj_fsub_s(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 1, 0, rm);
}

rvj_instr rvj_fmul_s(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 2, 0, rm);
}

rvj_instr rvj_fdiv_s(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 3, 0, rm);
}

rvj_instr rvj_fsqrt_s(REGF rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 0, 0xB, 0, rm);
}

rvj_instr rvj_fsgnj_s(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 4, 0, 0);
}

rvj_instr rvj_fsgnjn_s(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 4, 0, 1);
}

rvj_instr rvj_fsgnjx_s(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 4, 0, 2);
}

rvj_instr rvj_fmin_s(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 5, 0, 0);
}

rvj_instr rvj_fmax_s(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 5, 0, 1);
}

rvj_instr rvj_fcvt_w_s(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 0, 0x18, 0, rm);
}

rvj_instr rvj_fcvt_wu_s(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 1, 0x18, 0, rm);
}

rvj_instr rvj_fmv_x_w(REGX rd, REGF rs1) {
    return opFP(rd, rs1, 0, 0x1C, 0, 0);
}

rvj_instr rvj_feq_s(REGX rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 0x14, 0, 2);
}

rvj_instr rvj_flt_s(REGX rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 0x14, 0, 1);
}

rvj_instr rvj_fle_s(REGX rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 0x14, 0, 0);
}

rvj_instr rvj_fclass_s(REGX rd, REGF rs1) {
    return opFP(rd, rs1, 0, 0x1C, 0, 1);
}

rvj_instr rvj_fcvt_s_w(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 0, 0x1A, 0, rm);
}

rvj_instr rvj_fcvt_s_wu(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 1, 0x1A, 0, rm);
}

rvj_instr rvj_fmv_w_x(REGF rd, REGX rs1) {
    return opFP(rd, rs1, 0, 0x1E, 0, 0);
}

/******************************************************************************
** RV64-F Standard Extension
******************************************************************************/

rvj_instr rvj_fcvt_l_s(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 2, 0x18, 0, rm);
}

rvj_instr rvj_fcvt_lu_s(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 3, 0x18, 0, rm);
}

rvj_instr rvj_fcvt_s_l(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 2, 0x1A, 0, rm);
}

rvj_instr rvj_fcvt_s_lu(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 3, 0x1A, 0, rm);
}

/******************************************************************************
** RV32-D Standard Extension
******************************************************************************/

rvj_instr rvj_fld(REGF rd, REGX rs1, int offset) {
    return opI(rd, rs1, offset, 7, 3);
}

rvj_instr rvj_fsd(REGX rd, REGF rs1, int offset) {
    return opS(rd, rs1, offset, 0x27, 3);
}

rvj_instr rvj_fmadd_d(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x43, rm, 1);
}

rvj_instr rvj_fmsub_d(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x47, rm, 1);
}

rvj_instr rvj_fnmsub_d(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x4B, rm, 1);
}

rvj_instr rvj_fnmadd_d(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x4F, rm, 1);
}

rvj_instr rvj_fadd_d(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 0, 1, rm);
}

rvj_instr rvj_fsub_d(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 1, 1, rm);
}

rvj_instr rvj_fmul_d(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 2, 1, rm);
}

rvj_instr rvj_fdiv_d(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 3, 1, rm);
}

rvj_instr rvj_fsqrt_d(REGF rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 0, 0xB, 1, rm);
}

rvj_instr rvj_fsgnj_d(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 4, 1, 0);
}

rvj_instr rvj_fsgnjn_d(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 4, 1, 1);
}

rvj_instr rvj_fsgnjx_d(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 4, 1, 2);
}

rvj_instr rvj_fmin_d(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 5, 1, 0);
}

rvj_instr rvj_fmax_d(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 5, 1, 1);
}

rvj_instr rvj_fcvt_s_d(REGF rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 1, 0x8, 0, rm);
}

rvj_instr rvj_fcvt_d_s(REGF rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 0, 0x8, 1, rm);
}

rvj_instr rvj_feq_d(REGX rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 0x14, 1, 2);
}

rvj_instr rvj_flt_d(REGX rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 0x14, 1, 1);
}

rvj_instr rvj_fle_d(REGX rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 0x14, 1, 0);
}

rvj_instr rvj_fclass_d(REGX rd, REGF rs1) {
    return opFP(rd, rs1, 0, 0x1C, 1, 1);
}

rvj_instr rvj_fcvt_w_d(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 0, 0x18, 1, rm);
}

rvj_instr rvj_fcvt_wu_d(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 1, 0x18, 1, rm);
}

rvj_instr rvj_fcvt_d_w(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 0, 0x1A, 1, rm);
}

rvj_instr rvj_fcvt_d_wu(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 1, 0x1A, 1, rm);
}

/******************************************************************************
** RV64-D Standard Extension
******************************************************************************/

rvj_instr rvj_fcvt_l_d(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 2, 0x18, 1, rm);
}

rvj_instr rvj_fcvt_lu_d(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 3, 0x18, 1, rm);
}

rvj_instr rvj_fmv_x_d(REGX rd, REGF rs1) {
    return opFP(rd, rs1, 0, 0x1C, 1, 0);
}

rvj_instr rvj_fcvt_d_l(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 2, 0x1A, 1, rm);
}

rvj_instr rvj_fcvt_d_lu(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 3, 0x1A, 1, rm);
}

rvj_instr rvj_fmv_d_x(REGF rd, REGX rs1) {
    return opFP(rd, rs1, 0, 0x1E, 1, 0);
}

/******************************************************************************
** RV32-Q Standard Extension
******************************************************************************/

rvj_instr rvj_flq(REGF rd, REGX rs1, int offset) {
    return opI(rd, rs1, offset, 7, 4);
}

rvj_instr rvj_fsq(REGX rd, REGF rs1, int offset) {
    return opS(rd, rs1, offset, 0x27, 4);
}

rvj_instr rvj_fmadd_q(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x43, rm, 3);
}

rvj_instr rvj_fmsub_q(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x47, rm, 3);
}

rvj_instr rvj_fnmsub_q(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x4B, rm, 3);
}

rvj_instr rvj_fnmadd_q(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm) {
    return opR4(rd, rs1, rs2, rs3, 0x4F, rm, 3);
}

rvj_instr rvj_fadd_q(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 0, 3, rm);
}

rvj_instr rvj_fsub_q(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 1, 3, rm);
}

rvj_instr rvj_fmul_q(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 2, 3, rm);
}

rvj_instr rvj_fdiv_q(REGF rd, REGF rs1, REGF rs2, rvj_rm rm) {
    return opFP(rd, rs1, rs2, 3, 3, rm);
}

rvj_instr rvj_fsqrt_q(REGF rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 0, 0xB, 3, rm);
}

rvj_instr rvj_fsgnj_q(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 4, 3, 0);
}

rvj_instr rvj_fsgnjn_q(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 4, 3, 1);
}

rvj_instr rvj_fsgnjx_q(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 4, 3, 2);
}

rvj_instr rvj_fmin_q(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 5, 3, 0);
}

rvj_instr rvj_fmax_q(REGF rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 5, 3, 1);
}

rvj_instr rvj_fcvt_s_q(REGF rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 3, 8, 0, rm);
}

rvj_instr rvj_fcvt_q_s(REGF rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 0, 8, 3, rm);
}

rvj_instr rvj_fcvt_d_q(REGF rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 3, 8, 1, rm);
}

rvj_instr rvj_fcvt_q_d(REGF rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 1, 8, 3, rm);
}

rvj_instr rvj_feq_q(REGX rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 0x14, 3, 2);
}

rvj_instr rvj_flt_q(REGX rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 0x14, 3, 1);
}

rvj_instr rvj_fle_q(REGX rd, REGF rs1, REGF rs2) {
    return opFP(rd, rs1, rs2, 0x14, 3, 0);
}

rvj_instr rvj_fclass_q(REGX rd, REGF rs1) {
    return opFP(rd, rs1, 0, 0x1C, 3, 1);
}

rvj_instr rvj_fcvt_w_q(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 0, 0x18, 3, rm);
}

rvj_instr rvj_fcvt_wu_q(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 1, 0x18, 3, rm);
}

rvj_instr rvj_fcvt_q_w(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 0, 0x1A, 3, rm);
}

rvj_instr rvj_fcvt_q_wu(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 1, 0x1A, 3, rm);
}

/******************************************************************************
** RV64-Q Standard Extension
******************************************************************************/

rvj_instr rvj_fcvt_l_q(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 2, 0x18, 3, rm);
}

rvj_instr rvj_fcvt_lu_q(REGX rd, REGF rs1, rvj_rm rm) {
    return opFP(rd, rs1, 3, 0x18, 3, rm);
}

rvj_instr rvj_fcvt_q_l(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 2, 0x1A, 3, rm);
}

rvj_instr rvj_fcvt_q_lu(REGF rd, REGX rs1, rvj_rm rm) {
    return opFP(rd, rs1, 3, 0x1A, 3, rm);
}

/******************************************************************************
** RV64-V Standard Extension
******************************************************************************/

// Vector Control
rvj_instr rvj_vsetvl(REGX rd, REGX rs1, REGX rs2) {
    return opV(rd, rs1, rs2, 0x20, 0x7, 0x1);
}

rvj_instr rvj_vsetvli(REGX rd, REGX rs1, unsigned int vtypei) {
    return opV(rd, rs1, (vtypei & 0x1F),
        ((vtypei >> 6) & 0x1F), 0x7, (vtypei & 0x20));
}

rvj_instr rvj_vsetivli(REGX rd, unsigned int uimm, unsigned int vtypei) {
    return opV(rd, uimm, (vtypei & 0x1F),
        ((0x3 << 4) | ((vtypei >> 6) & 0xF)), 0x7, (vtypei & 0x20));
}

// Vector Load (unit-stride)
rvj_instr rvj_vl(REGV vd, REGX rs1, rvj_vmask vm, unsigned int eew_bytes) {
    switch (eew_bytes) {
        case 1: return rvj_vle8(vd, rs1, vm);
        case 2: return rvj_vle16(vd, rs1, vm);
        case 4: return rvj_vle32(vd, rs1, vm);
        case 8: return rvj_vle64(vd, rs1, vm);
        case 16: return rvj_vle128(vd, rs1, vm);
        case 32: return rvj_vle256(vd, rs1, vm);
        case 64: return rvj_vle512(vd, rs1, vm);
        case 128: return rvj_vle1024(vd, rs1, vm);
        default: return 0;
    }
}

rvj_instr rvj_vle8(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5), 0x7, 0x0);
}

rvj_instr rvj_vle16(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5), 0x7, 0x5);
}

rvj_instr rvj_vle32(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5), 0x7, 0x6);
}

rvj_instr rvj_vle64(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5), 0x7, 0x7);
}

rvj_instr rvj_vle128(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, (0x100 | (vm & 1) << 5), 0x7, 0x0);
}

rvj_instr rvj_vle256(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, (0x100 | (vm & 1) << 5), 0x7, 0x5);
}

rvj_instr rvj_vle512(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, (0x100 | (vm & 1) << 5), 0x7, 0x6);
}

rvj_instr rvj_vle1024(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, (0x100 | (vm & 1) << 5), 0x7, 0x7);
}

// Vector Store (unit-stride)
rvj_instr rvj_vs(REGV vs3, REGX rs1, rvj_vmask vm, unsigned int eew_bytes) {
    switch (eew_bytes) {
        case 1: return rvj_vse8(vs3, rs1, vm);
        case 2: return rvj_vse16(vs3, rs1, vm);
        case 4: return rvj_vse32(vs3, rs1, vm);
        case 8: return rvj_vse64(vs3, rs1, vm);
        case 16: return rvj_vse128(vs3, rs1, vm);
        case 32: return rvj_vse256(vs3, rs1, vm);
        case 64: return rvj_vse512(vs3, rs1, vm);
        case 128: return rvj_vse1024(vs3, rs1, vm);
        default: return 0;
    }
}

rvj_instr rvj_vse8(REGV vs3, REGX rs1, rvj_vmask vm) {
    return opI(vs3, rs1, ((vm & 1) << 5), 0x27, 0x0);
}

rvj_instr rvj_vse16(REGV vs3, REGX rs1, rvj_vmask vm) {
    return opI(vs3, rs1, ((vm & 1) << 5), 0x27, 0x5);
}

rvj_instr rvj_vse32(REGV vs3, REGX rs1, rvj_vmask vm) {
    return opI(vs3, rs1, ((vm & 1) << 5), 0x27, 0x6);
}

rvj_instr rvj_vse64(REGV vs3, REGX rs1, rvj_vmask vm) {
    return opI(vs3, rs1, ((vm & 1) << 5), 0x27, 0x7);
}

rvj_instr rvj_vse128(REGV vs3, REGX rs1, rvj_vmask vm) {
    return opI(vs3, rs1, (0x100 | (vm & 1) << 5), 0x27, 0x0);
}

rvj_instr rvj_vse256(REGV vs3, REGX rs1, rvj_vmask vm) {
    return opI(vs3, rs1, (0x100 | (vm & 1) << 5), 0x27, 0x5);
}

rvj_instr rvj_vse512(REGV vs3, REGX rs1, rvj_vmask vm) {
    return opI(vs3, rs1, (0x100 | (vm & 1) << 5), 0x27, 0x6);
}

rvj_instr rvj_vse1024(REGV vs3, REGX rs1, rvj_vmask vm) {
    return opI(vs3, rs1, (0x100 | (vm & 1) << 5), 0x27, 0x7);
}

// Vector Load (constant stride)
rvj_instr rvj_vls(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm, unsigned int eew_bytes) {
    switch (eew_bytes) {
        case 1: return rvj_vlse8(vd, rs1, rs2, vm);
        case 2: return rvj_vlse16(vd, rs1, rs2, vm);
        case 4: return rvj_vlse32(vd, rs1, rs2, vm);
        case 8: return rvj_vlse64(vd, rs1, rs2, vm);
        case 16: return rvj_vlse128(vd, rs1, rs2, vm);
        case 32: return rvj_vlse256(vd, rs1, rs2, vm);
        case 64: return rvj_vlse512(vd, rs1, rs2, vm);
        case 128: return rvj_vlse1024(vd, rs1, rs2, vm);
        default: return 0;
    }
}

rvj_instr rvj_vlse8(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x80 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x7, 0x0);
}

rvj_instr rvj_vlse16(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x80 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x7, 0x5);
}

rvj_instr rvj_vlse32(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x80 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x7, 0x6);
}

rvj_instr rvj_vlse64(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x80 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x7, 0x7);
}

rvj_instr rvj_vlse128(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x180 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x7, 0x0);
}

rvj_instr rvj_vlse256(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x180 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x7, 0x5);
}

rvj_instr rvj_vlse512(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x180 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x7, 0x6);
}

rvj_instr rvj_vlse1024(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x180 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x7, 0x7);
}

// Vector Store (constant stride)
rvj_instr rvj_vss(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm, unsigned int eew_bytes) {
    switch (eew_bytes) {
        case 1: return rvj_vsse8(vs3, rs1, rs2, vm);
        case 2: return rvj_vsse16(vs3, rs1, rs2, vm);
        case 4: return rvj_vsse32(vs3, rs1, rs2, vm);
        case 8: return rvj_vsse64(vs3, rs1, rs2, vm);
        case 16: return rvj_vsse128(vs3, rs1, rs2, vm);
        case 32: return rvj_vsse256(vs3, rs1, rs2, vm);
        case 64: return rvj_vsse512(vs3, rs1, rs2, vm);
        case 128: return rvj_vsse1024(vs3, rs1, rs2, vm);
        default: return 0;
    }
}

rvj_instr rvj_vsse8(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x80 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x27, 0x0);
}

rvj_instr rvj_vsse16(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x80 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x27, 0x5);
}

rvj_instr rvj_vsse32(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x80 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x27, 0x6);
}

rvj_instr rvj_vsse64(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x80 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x27, 0x7);
}

rvj_instr rvj_vsse128(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x180 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x27, 0x0);
}

rvj_instr rvj_vsse256(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x180 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x27, 0x5);
}

rvj_instr rvj_vsse512(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x180 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x27, 0x6);
}

rvj_instr rvj_vsse1024(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x180 | ((vm & 1) << 5) | (rs2 & 0x1F), 0x27, 0x7);
}

// Vector Load (unordered indexed)
rvj_instr rvj_vlux(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm, unsigned int eew_bytes) {
    switch (eew_bytes) {
        case 1: return rvj_vluxei8(vd, rs1, vs2, vm);
        case 2: return rvj_vluxei16(vd, rs1, vs2, vm);
        case 4: return rvj_vluxei32(vd, rs1, vs2, vm);
        case 8: return rvj_vluxei64(vd, rs1, vs2, vm);
        case 16: return rvj_vluxei128(vd, rs1, vs2, vm);
        case 32: return rvj_vluxei256(vd, rs1, vs2, vm);
        case 64: return rvj_vluxei512(vd, rs1, vs2, vm);
        case 128: return rvj_vluxei1024(vd, rs1, vs2, vm);
        default: return 0;
    }
}

rvj_instr rvj_vluxei8(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x40 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x0);
}

rvj_instr rvj_vluxei16(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x40 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x5);
}

rvj_instr rvj_vluxei32(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x40 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x6);
}

rvj_instr rvj_vluxei64(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x40 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x7);
}

rvj_instr rvj_vluxei128(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x140 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x0);
}

rvj_instr rvj_vluxei256(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x140 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x5);
}

rvj_instr rvj_vluxei512(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x140 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x6);
}

rvj_instr rvj_vluxei1024(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x140 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x7);
}

// Vector Store (unordered indexed)
rvj_instr rvj_vsux(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm, unsigned int eew_bytes) {
    switch (eew_bytes) {
        case 1: return rvj_vsuxei8(vs3, rs1, vs2, vm);
        case 2: return rvj_vsuxei16(vs3, rs1, vs2, vm);
        case 4: return rvj_vsuxei32(vs3, rs1, vs2, vm);
        case 8: return rvj_vsuxei64(vs3, rs1, vs2, vm);
        case 16: return rvj_vsuxei128(vs3, rs1, vs2, vm);
        case 32: return rvj_vsuxei256(vs3, rs1, vs2, vm);
        case 64: return rvj_vsuxei512(vs3, rs1, vs2, vm);
        case 128: return rvj_vsuxei1024(vs3, rs1, vs2, vm);
        default: return 0;
    }
}

rvj_instr rvj_vsuxei8(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x40 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x0);
}

rvj_instr rvj_vsuxei16(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x40 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x5);
}

rvj_instr rvj_vsuxei32(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x40 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x6);
}

rvj_instr rvj_vsuxei64(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x40 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x7);
}

rvj_instr rvj_vsuxei128(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x140 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x0);
}

rvj_instr rvj_vsuxei256(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x140 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x5);
}

rvj_instr rvj_vsuxei512(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x140 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x6);
}

rvj_instr rvj_vsuxei1024(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x140 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x7);
}

// Vector Load (ordered indexed)
rvj_instr rvj_vlox(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm, unsigned int eew_bytes) {
    switch (eew_bytes) {
        case 1: return rvj_vloxei8(vd, rs1, vs2, vm);
        case 2: return rvj_vloxei16(vd, rs1, vs2, vm);
        case 4: return rvj_vloxei32(vd, rs1, vs2, vm);
        case 8: return rvj_vloxei64(vd, rs1, vs2, vm);
        case 16: return rvj_vloxei128(vd, rs1, vs2, vm);
        case 32: return rvj_vloxei256(vd, rs1, vs2, vm);
        case 64: return rvj_vloxei512(vd, rs1, vs2, vm);
        case 128: return rvj_vloxei1024(vd, rs1, vs2, vm);
        default: return 0;
    }
}
rvj_instr rvj_vloxei8(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0xC0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x0);
}

rvj_instr rvj_vloxei16(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0xC0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x5);
}

rvj_instr rvj_vloxei32(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0xC0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x6);
}

rvj_instr rvj_vloxei64(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0xC0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x7);
}

rvj_instr rvj_vloxei128(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x1C0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x0);
}

rvj_instr rvj_vloxei256(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x1C0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x5);
}

rvj_instr rvj_vloxei512(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x1C0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x6);
}

rvj_instr rvj_vloxei1024(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vd, rs1, 0x1C0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x7, 0x7);
}

// Vector Store (ordered indexed)
rvj_instr rvj_vsox(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm, unsigned int eew_bytes) {
    switch (eew_bytes) {
        case 1: return rvj_vsoxei8(vs3, rs1, vs2, vm);
        case 2: return rvj_vsoxei16(vs3, rs1, vs2, vm);
        case 4: return rvj_vsoxei32(vs3, rs1, vs2, vm);
        case 8: return rvj_vsoxei64(vs3, rs1, vs2, vm);
        case 16: return rvj_vsoxei128(vs3, rs1, vs2, vm);
        case 32: return rvj_vsoxei256(vs3, rs1, vs2, vm);
        case 64: return rvj_vsoxei512(vs3, rs1, vs2, vm);
        case 128: return rvj_vsoxei1024(vs3, rs1, vs2, vm);
        default: return 0;
    }
}

rvj_instr rvj_vsoxei8(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0xC0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x0);
}

rvj_instr rvj_vsoxei16(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0xC0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x5);
}

rvj_instr rvj_vsoxei32(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0xC0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x6);
}

rvj_instr rvj_vsoxei64(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0xC0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x7);
}

rvj_instr rvj_vsoxei128(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x1C0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x0);
}

rvj_instr rvj_vsoxei256(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x1C0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x5);
}

rvj_instr rvj_vsoxei512(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x1C0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x6);
}

rvj_instr rvj_vsoxei1024(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm) {
    return opI(vs3, rs1, 0x1C0 | ((vm & 1) << 5) | (vs2 & 0x1F), 0x27, 0x7);
}

// Vector Unit-stride Fault-Only-First Loads
rvj_instr rvj_vlff(REGV vd, REGX rs1, rvj_vmask vm, unsigned int eew_bytes) {
    switch (eew_bytes) {
        case 1: return rvj_vle8ff(vd, rs1, vm);
        case 2: return rvj_vle16ff(vd, rs1, vm);
        case 4: return rvj_vle32ff(vd, rs1, vm);
        case 8: return rvj_vle64ff(vd, rs1, vm);
        case 16: return rvj_vle128ff(vd, rs1, vm);
        case 32: return rvj_vle256ff(vd, rs1, vm);
        case 64: return rvj_vle512ff(vd, rs1, vm);
        case 128: return rvj_vle1024ff(vd, rs1, vm);
        default: return 0;
    }
}

rvj_instr rvj_vle8ff(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5) | 0x10, 0x7, 0x0);
}

rvj_instr rvj_vle16ff(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5) | 0x10, 0x7, 0x5);
}

rvj_instr rvj_vle32ff(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5) | 0x10, 0x7, 0x6);
}

rvj_instr rvj_vle64ff(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5) | 0x10, 0x7, 0x7);
}

rvj_instr rvj_vle128ff(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5) | 0x110, 0x7, 0x0);
}

rvj_instr rvj_vle256ff(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5) | 0x110, 0x7, 0x5);
}

rvj_instr rvj_vle512ff(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5) | 0x110, 0x7, 0x6);
}

rvj_instr rvj_vle1024ff(REGV vd, REGX rs1, rvj_vmask vm) {
    return opI(vd, rs1, ((vm & 1) << 5) | 0x110, 0x7, 0x7);
}

// Vector Load/Store Segment Instructions (TODO)
// Vector Load/Store Whole Register Instructions (TODO)
// Vector AMO Instructions (TODO)
// Vector Integer Arithmetic Instructions (TODO)
#define def_IVV(NAME, F6)\
rvj_instr rvj_##NAME##_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm) {\
    return opIVV(vd, vs1, vs2, F6, vm);\
}

#define def_IVX(NAME, F6)\
rvj_instr rvj_##NAME##_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {\
    return opIVX(vd, rs1, vs2, F6, vm);\
}

#define def_IVI(NAME, F6)\
rvj_instr rvj_##NAME##_vi(REGV vd, REGV vs2, int simm5, rvj_vmask vm) {\
    return opIVI(vd, simm5, vs2, F6, vm);\
}

#define def_MVV(NAME, F6)\
rvj_instr rvj_##NAME##_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm) {\
    return opMVV(vd, vs1, vs2, F6, vm);\
}

#define def_MVX(NAME, F6)\
rvj_instr rvj_##NAME##_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm) {\
    return opMVX(vd, rs1, vs2, F6, vm);\
}

#define def_FVV(NAME, F6)\
rvj_instr rvj_##NAME##_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm) {\
    return opFVV(vd, vs1, vs2, F6, vm);\
}

#define def_FVF(NAME, F6)\
rvj_instr rvj_##NAME##_vf(REGV vd, REGF rs1, REGV vs2, rvj_vmask vm) {\
    return opFVF(vd, rs1, vs2, F6, vm);\
}

#define def_IV_VX(NAME, F6) def_IVV(NAME, F6) def_IVX(NAME, F6)
#define def_IV_VXI(NAME, F6) def_IV_VX(NAME, F6) def_IVI(NAME, F6)
#define def_MV_VX(NAME, F6) def_MVV(NAME, F6) def_MVX(NAME, F6)
#define def_FV_VF(NAME, F6) def_FVV(NAME, F6) def_FVF(NAME, F6)

// Vector Integer Add
def_IV_VXI(vadd, 0x0)

// Vector Integer Subtraction
def_IV_VX(vsub, 0x2)

// Vector Integer Reverse Subtraction
def_IVX(vrsub, 0x3)
def_IVI(vrsub, 0x3)

// Vector Bitwise AND
def_IV_VXI(vand, 0x9)

// Vector Bitwise OR
def_IV_VXI(vor, 0x10)

// Vector Bitwise XOR
def_IV_VXI(vxor, 0x0B)

// Vector Bitwise SLT
def_IV_VXI(vslt, 0x1B)

// Vector Bitwise SEQ
def_IV_VXI(vseq, 0x18)

// Vector Bitwise SGT
def_IV_VXI(vsgt, 0x1f)

// Vector Mask AND
def_IV_VXI(vmand, 0x19)

// Vector Mask OR
def_IV_VXI(vmor, 0x1a)

// Vector Integer Divide
def_IV_VX(vdiv, 0x21)

// Vector Integer Multiply
def_IV_VX(vmul, 0x25)

// Vector Floating-point Fused Multiply and Add (vd is first multiplicand)
def_MV_VX(vmadd, 0x29)

// Vector Floating-point Fused Multiply and Add (vd is addend)
def_MV_VX(vmacc, 0x2D)

// Vector Integer Merge
rvj_instr rvj_vmerge_vvm(REGV vd, REGV vs2, REGV vs1) {
    return opIVV(vd, vs1, vs2, 0x17, rvj_masked);
}

rvj_instr rvj_vmerge_vxm(REGV vd, REGV vs2, REGX rs1) {
    return opIVX(vd, rs1, vs2, 0x17, rvj_masked);
}

rvj_instr rvj_vmerge_vim(REGV vd, REGV vs2, int simm5) {
    return opIVI(vd, simm5, vs2, 0x17, rvj_masked);
}

// Vector Integer Move
rvj_instr rvj_vmv_vv(REGV vd, REGV vs1) {
    return opIVV(vd, vs1, 0, 0x17, rvj_unmasked);
}
rvj_instr rvj_vmv_vx(REGV vd, REGX rs1) {
    return opIVX(vd, rs1, 0, 0x17, rvj_unmasked);
}
rvj_instr rvj_vmv_vi(REGV vd, int simm5) {
    return opIVI(vd, simm5, 0, 0x17, rvj_unmasked);
}

// Vector Fixed-Point Arithmetic Instructions (TODO)
// Vector Floating-Point Arithmetic Operations (TODO)

// Vector Floating-point Add
def_FV_VF(vfadd, 0x0)

// Vector Floating-point Subtraction
def_FV_VF(vfsub, 0x2)

// Vector Floating-point Reverse Subtraction
def_FVF(vfrsub, 0x27)

// Vector Floating-point Divide
def_FV_VF(vfdiv, 0x20)

// Vector Floating-point Reverse Divide
def_FVF(vfrdiv, 0x21)

// Vector Floating-point Multiply
def_FV_VF(vfmul, 0x24)

// Vector Floating-point Fused Multiply and Add (vd is first multiplicand)
def_FV_VF(vfmadd, 0x28)

// Vector Floating-point Fused Multiply and Add (vd is addend)
def_FV_VF(vfmacc, 0x2C)

// Vector Reduction Operations (TODO)
// Vector Mask Instructions (TODO)
rvj_instr rvj_vmand_mm(REGV vd, REGV vs1, REGV vs2) {
    return opMVV(vd, vs1, vs2, 0x19, rvj_unmasked);
}

rvj_instr rvj_vmor_mm(REGV vd, REGV vs1, REGV vs2) {
    return opMVV(vd, vs1, vs2, 0x1A, rvj_unmasked);
}

rvj_instr rvj_vmslt_vx(REGV vd, REGV vs1, REGV vs2) {
    return opIVX(vd, vs1, vs2, 0x1B, rvj_unmasked);
}

rvj_instr rvj_vmseq_vv(REGV vd, REGV vs1, REGV vs2) {
    return opIVV(vd, vs1, vs2, 0x18, rvj_unmasked);
}

rvj_instr rvj_vmsgt_vx(REGV vd, REGV vs1, REGV vs2) {
    return opIVX(vd, vs1, vs2, 0x1f, rvj_unmasked);
}
// Vector Permutation Instructions (TODO)
rvj_instr rvj_vslideup_vx(REGV vd, REGV vs2, REGX rs1) {
    return opIVX(vd, vs2, rs1, 0x0e, rvj_unmasked);
}
// Vector Integer Scalar Move Instructions
rvj_instr rvj_vmv_xs(REGX vd, REGV vs2) {
    return opMVV(vd, 0, vs2, 0x10, rvj_unmasked);
}

rvj_instr rvj_vmv_sx(REGV vd, REGX rs1) {
    return opMVX(vd, rs1, 0, 0x10, rvj_unmasked);
}

// Vector Floating-Point Scalar Move Instructions
rvj_instr rvj_vfmv_fs(REGF vd, REGV vs2) {
    return opFVV(vd, 0, vs2, 0x10, rvj_unmasked);
}

rvj_instr rvj_vfmv_sf(REGV vd, REGF rs1) {
    return opFVF(vd, rs1, 0, 0x10, rvj_unmasked);
}

// TODO ...

// Whole Vector Register Move
rvj_instr rvj_vmvr_v(REGV vd, REGV vs2, unsigned int nr) {
    switch (nr) {
        case 1: return rvj_vmv1r_v(vd, vs2);
        case 2: return rvj_vmv2r_v(vd, vs2);
        case 4: return rvj_vmv4r_v(vd, vs2);
        case 8: return rvj_vmv8r_v(vd, vs2);
        default: return 0;
    }
}

rvj_instr rvj_vmv1r_v(REGV vd, REGV vs2) {
    return opIVI(vd, 0, vs2, 0x27, rvj_unmasked);
}

rvj_instr rvj_vmv2r_v(REGV vd, REGV vs2) {
    return opIVI(vd, 1, vs2, 0x27, rvj_unmasked);
}

rvj_instr rvj_vmv4r_v(REGV vd, REGV vs2) {
    return opIVI(vd, 3, vs2, 0x27, rvj_unmasked);
}

rvj_instr rvj_vmv8r_v(REGV vd, REGV vs2) {
    return opIVI(vd, 7, vs2, 0x27, rvj_unmasked);
}


#undef def_IVV
#undef def_IVX
#undef def_IVI
#undef def_MVV
#undef def_MVX
#undef def_FVV
#undef def_FVF
#undef def_IV_VX
#undef def_IV_VXI
#undef def_MV_VX
#undef def_FV_VF
/******************************************************************************
** RV64-V Standard Extension Pseudo Instructions
******************************************************************************/

rvj_instr rvj_vsetvlmax(REGX rd, REGX rs2) {
    return rvj_vsetvl(rd, 0, rs2);
}

rvj_instr rvj_vsetvlmaxi(REGX rd, unsigned int vtypei) {
    return rvj_vsetvli(rd, 0, vtypei);
}

#undef REGX
#undef REGF
#undef REGV