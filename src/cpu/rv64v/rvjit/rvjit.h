#ifndef RVJIT_H
#define RVJIT_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef RVJ_VSPEC_0_7
    #undef RVJ_VSPEC_1_0
#else
    #define RVJ_VSPEC_1_0
#endif

/// @brief RISC-V JIT assembler opaque structure
struct rvj_asm;

/// @brief RISC-V JIT function opaque structure
struct rvj_function;

/// @brief RISC-V binary instruction
typedef unsigned int rvj_instr;

/// @brief RISC-V register bit field
typedef unsigned int rvj_reg;

/// @brief RISC-V JIT assembler handle
typedef struct rvj_asm* rvj_asm_t;

/// @brief RISC-V JIT function handle
typedef struct rvj_function* rvj_function_t;

/// @brief Status code for API functions
typedef enum {
    rvj_success,           /// Success
    rvj_invalid_arguments, /// Invalid user inputs
    rvj_empty,             /// Code Region is empty
    rvj_out_of_memory,     /// Failure to allocate memory
    rvj_error,             /// Internal error (bug)
    rvj_undefined_label    /// Label used but not defined
} rvj_status_t;

/// @brief Type that represents a RISC-V named general purpose register
typedef enum {
    rvj_x0 = 0,
    rvj_ra = 1,
    rvj_sp = 2,
    rvj_gp = 3,
    rvj_tp = 4,
    rvj_t0 = 5,
    rvj_t1 = 6,
    rvj_t2 = 7,
    rvj_s0 = 8,
    rvj_s1 = 9,
    rvj_a0 = 10,
    rvj_a1 = 11,
    rvj_a2 = 12,
    rvj_a3 = 13,
    rvj_a4 = 14,
    rvj_a5 = 15,
    rvj_a6 = 16,
    rvj_a7 = 17,
    rvj_s2 = 18,
    rvj_s3 = 19,
    rvj_s4 = 20,
    rvj_s5 = 21,
    rvj_s6 = 22,
    rvj_s7 = 23,
    rvj_s8 = 24,
    rvj_s9 = 25,
    rvj_s10 = 26,
    rvj_s11 = 27,
    rvj_t3 = 28,
    rvj_t4 = 29,
    rvj_t5 = 30,
    rvj_t6 = 31
} rvj_gpr;

/// @brief Type that represents a RISC-V named floating-point register
typedef enum {
    rvj_ft0 = 0,
    rvj_ft1 = 1,
    rvj_ft2 = 2,
    rvj_ft3 = 3,
    rvj_ft4 = 4,
    rvj_ft5 = 5,
    rvj_ft6 = 6,
    rvj_ft7 = 7,
    rvj_fs0 = 8,
    rvj_fs1 = 9,
    rvj_fa0 = 10,
    rvj_fa1 = 11,
    rvj_fa2 = 12,
    rvj_fa3 = 13,
    rvj_fa4 = 14,
    rvj_fa5 = 15,
    rvj_fa6 = 16,
    rvj_fa7 = 17,
    rvj_fs2 = 18,
    rvj_fs3 = 19,
    rvj_fs4 = 20,
    rvj_fs5 = 21,
    rvj_fs6 = 22,
    rvj_fs7 = 23,
    rvj_fs8 = 24,
    rvj_fs9 = 25,
    rvj_fs10 = 26,
    rvj_fs11 = 27,
    rvj_ft8 = 28,
    rvj_ft9 = 29,
    rvj_ft10 = 30,
    rvj_ft11 = 31
} rvj_fpr;

/// @brief Type that represents a RISC-V named vector register
typedef enum {
    rvj_v0 = 0,
    rvj_v1 = 1,
    rvj_v2 = 2,
    rvj_v3 = 3,
    rvj_v4 = 4,
    rvj_v5 = 5,
    rvj_v6 = 6,
    rvj_v7 = 7,
    rvj_v8 = 8,
    rvj_v9 = 9,
    rvj_v10 = 10,
    rvj_v11 = 11,
    rvj_v12 = 12,
    rvj_v13 = 13,
    rvj_v14 = 14,
    rvj_v15 = 15,
    rvj_v16 = 16,
    rvj_v17 = 17,
    rvj_v18 = 18,
    rvj_v19 = 19,
    rvj_v20 = 20,
    rvj_v21 = 21,
    rvj_v22 = 22,
    rvj_v23 = 23,
    rvj_v24 = 24,
    rvj_v25 = 25,
    rvj_v26 = 26,
    rvj_v27 = 27,
    rvj_v28 = 28,
    rvj_v29 = 29,
    rvj_v30 = 30,
    rvj_v31 = 31
} rvj_vr;

/// @brief Rounding modes supported on floating-points operations
/// @details
/// Floating-point operations use either a static rounding mode encoded in the
/// instruction, or a dynamic rounding mode held in `frm' (CSR register).
/// A value of 0x111 (7) in the instruction's rm field selects the dynamic
/// rounding mode held in `frm'.
typedef enum {
    rvj_rne = 0, // Round to Nearest, ties to Even
    rvj_rtz = 1, // Round towards Zero
    rvj_rdn = 2, // Round Down (towards − infinity)
    rvj_rup = 3, // Round Up (towards + infinity)
    rvj_rmm = 4, // Round to Nearest, ties to Max Magnitude
    rvj_dyn = 7, // Selects dynamic rounding mode
} rvj_rm;

/// @brief Bitmask for composing the contents of the vtype field for the
/// V standard extension vsetvl instructions
/// @details Combine values with bitwise OR operator, the combining the mask
/// policy (vma), the tail policy (vta), single-element width (e8-e1024),
/// and length multiplier (mf8-m8).
typedef enum {
#ifdef RVJ_VSPEC_1_0
    rvj_e8 = 0,       // Set single element width size to 8 bytes
    rvj_e16 = 0x8,    // Set single element width size to 16 bytes
    rvj_e32 = 0x10,   // Set single element width size to 32 bytes
    rvj_e64 = 0x18,   // Set single element width size to 64 bytes
    rvj_e128 = 0x20,  // Set single element width size to 128 bytes
    rvj_e256 = 0x28,  // Reserved but unsupported officially
    rvj_e512 = 0x30,  // Reserved but unsupported officially
    rvj_e1024 = 0x38, // Reserved but unsupported officially
    rvj_vma = 0x80,   // Set mask-policy to 'vector mask agnostic'
    rvj_vta = 0x40,   // Set tail-policy to 'vector tail agnostic'
    rvj_m1 = 0x0,     // Set the length multiplier to 1
    rvj_m2 = 0x1,     // Set the length multiplier to 2
    rvj_m4 = 0x2,     // Set the length multiplier to 4
    rvj_m8 = 0x3,     // Set the length multiplier to 8
    rvj_mf8 = 0x5,    // Set the length multiplier to 1/8
    rvj_mf4 = 0x6,    // Set the length multiplier to 1/4
    rvj_mf2 = 0x7     // Set the length multiplier to 1/2
#else
    rvj_e8 = 0,       // Set single element width size to 8 bytes
    rvj_e16 = 0x4,    // Set single element width size to 16 bytes
    rvj_e32 = 0x8,    // Set single element width size to 32 bytes
    rvj_e64 = 0xC,    // Set single element width size to 64 bytes
    rvj_e128 = 0x10,  // Set single element width size to 128 bytes
    rvj_e256 = 0x14,  // Reserved but unsupported officially
    rvj_e512 = 0x18,  // Reserved but unsupported officially
    rvj_e1024 = 0x1C, // Reserved but unsupported officially
    rvj_m1 = 0x0,     // Set the length multiplier to 1
    rvj_m2 = 0x1,     // Set the length multiplier to 2
    rvj_m4 = 0x2,     // Set the length multiplier to 4
    rvj_m8 = 0x3      // Set the length multiplier to 8
#endif
} rvj_vtype_mask;

/// @brief Flag to determines if vector instructions use a mask register
/// @details When unmasked, the vector instructions operate on all vector
/// register elements.
typedef enum {
    rvj_unmasked = 0x1,
    rvj_masked = 0x0,
} rvj_vmask;

/// @brief Initialize the JIT Assembler
/// @param h The JIT assembler handle
/// @param version The RISC-V V specification version
/// @details The name argument defines the filename when dumping the assembly
rvj_status_t rvj_asm_init(rvj_asm_t *h);

/// @brief Deallocate the resources used by the JIT Assembler
/// @param h The JIT assembler handle
/// @details Does not deallocate functions created by the assembler
void rvj_asm_free(rvj_asm_t *h);

/// @brief Add a label to the JIT context at the current instruction
/// @param h The JIT assembler handle
/// @param name The label name identifier (up to 28 characters)
rvj_status_t rvj_asm_label(rvj_asm_t h, const char* name);

/// @brief Append an instruction to the JIT stream
/// @param h The JIT assembler handle
/// @param w The instruction in binary format
rvj_status_t rvj_asm_push(rvj_asm_t h, rvj_instr w);

/// @brief Append a branch or jump to the JIT stream that references a label
/// @param h The JIT assembler handle
/// @param w The instruction in binary format
/// @param name The label name identifier (up to 28 characters)
/// @details This API call is intended for use with branches, enabling the
/// assembler to calculate memory offsets based on the label distance
rvj_status_t rvj_asm_push_lref(rvj_asm_t h, rvj_instr w, const char *name);

/// @brief Resolves label indices and prepares the function handle
/// @param h The JIT assembler handle
/// @details The function cannot be changed after it has been completed
/// @details This function interfaces with the OS to mark memory locations
/// as executable and can be executed only once per JIT assembler handle
rvj_status_t rvj_asm_done(rvj_asm_t h);

/// @brief Obtain the function handle created by the JIT assembler
/// @param h The JIT assembler handle
/// @param out The JIT function handle
rvj_status_t rvj_asm_get_function_handle(rvj_asm_t h,
                                         rvj_function_t *out);

/// @brief Obtain the size in bytes for the compiled function
/// @param h The JIT assembler handle
size_t rvj_function_get_size(const rvj_function_t h);

/// @brief Dump the function binary to the out buffer
/// @param h The JIT assembler handle
/// @param out The output byte buffer
/// @details The buffer must be large enough to contain the function
/// @see rvj_function_get_size
size_t rvj_function_dump(const rvj_function_t h, char * const out);

/// @brief Obtain the function pointer from a JIT function handle
/// @param h The JIT function handle
/// @param out The JIT assembled function pointer
void rvj_function_get_pointer(const rvj_function_t h, void **out);

/// @brief Deallocates and destroys the function created by a JIT assembler
/// @param h The JIT function handle
/// @details This function will unmap executable memory locations previously
/// allocated with @ref rvj_asm_done.
void rvj_function_free(rvj_function_t *h);

/// @brief Obtains the register id of a named RISC-V general purpose register
/// @param name The register mnemonic name
/// @param out The output register id
rvj_status_t rvj_get_gpr_id(const char * const name, rvj_gpr *out);

/// @brief Obtains the register id of a named RISC-V floating-point register
/// @param name The register mnemonic name
/// @param out The output register id
rvj_status_t rvj_get_fpr_id(const char * const name, rvj_fpr *out);

/// @brief Obtains the register id of a named RISC-V vector register
/// @param name The register mnemonic name
/// @param out The output register id
rvj_status_t rvj_get_vr_id(const char * const name, rvj_vr *out);

/******************************************************************************
** RV32-I Base Instruction Set
******************************************************************************/
#define REGX rvj_gpr // We use this to compact the list and undefine afterwards
#define REGF rvj_fpr // We use this to compact the list and undefine afterwards
#define REGV rvj_vr // We use this to compact the list and undefine afterwards

rvj_instr rvj_lui(REGX rd, int imm);
rvj_instr rvj_auipc(REGX rd, int imm);
rvj_instr rvj_jal(REGX rd, int offset);
rvj_instr rvj_jalr(REGX rd, REGX rs1, int offset);
rvj_instr rvj_beq(REGX rs1, REGX rs2, int offset);
rvj_instr rvj_bne(REGX rs1, REGX rs2, int offset);
rvj_instr rvj_blt(REGX rs1, REGX rs2, int offset);
rvj_instr rvj_bge(REGX rs1, REGX rs2, int offset);
rvj_instr rvj_bltu(REGX rs1, REGX rs2, int offset);
rvj_instr rvj_bgeu(REGX rs1, REGX rs2, int offset);
rvj_instr rvj_lb(REGX rd, REGX rs1, int offset);
rvj_instr rvj_lh(REGX rd, REGX rs1, int offset);
rvj_instr rvj_lw(REGX rd, REGX rs1, int offset);
rvj_instr rvj_lbu(REGX rd, REGX rs1, int offset);
rvj_instr rvj_lhu(REGX rd, REGX rs1, int offset);
rvj_instr rvj_sb(REGX src, REGX base, int offset);
rvj_instr rvj_sh(REGX src, REGX base, int offset);
rvj_instr rvj_sw(REGX src, REGX base, int offset);
rvj_instr rvj_addi(REGX rd, REGX rs1, int imm);
rvj_instr rvj_slti(REGX rd, REGX rs1, int imm);
rvj_instr rvj_sltiu(REGX rd, REGX rs1, int imm);
rvj_instr rvj_xori(REGX rd, REGX rs1, int imm);
rvj_instr rvj_ori(REGX rd, REGX rs1, int imm);
rvj_instr rvj_andi(REGX rd, REGX rs1, int imm);
rvj_instr rvj_slli(REGX rd, REGX rs1, int shamt);
rvj_instr rvj_srli(REGX rd, REGX rs1, int shamt);
rvj_instr rvj_srai(REGX rd, REGX rs1, int shamt);
rvj_instr rvj_add(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_sub(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_sll(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_slt(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_sltu(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_xor_(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_srl(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_sra(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_or_(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_and_(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_fence(REGX rd, REGX rs1, int imm);
rvj_instr rvj_ecall();
rvj_instr rvj_ebreak();
/******************************************************************************
** RV32-I Pseudo Instructions
******************************************************************************/
rvj_instr rvj_nop();
rvj_instr rvj_ret();
rvj_instr rvj_mv(REGX rd, REGX rs1);
rvj_instr rvj_li(REGX rd, const int imm);
rvj_instr rvj_j(const int imm);
rvj_instr rvj_beqz(REGX rd, int offset);
rvj_instr rvj_bnez(REGX rd, int offset);
rvj_instr rvj_sext_w(REGX rd, REGX rs1);
/******************************************************************************
** RV64-I Standard Extension
******************************************************************************/
rvj_instr rvj_lwu(REGX rd, REGX rs1, int offset);
rvj_instr rvj_ld(REGX rd, REGX rs1, int offset);
rvj_instr rvj_sd(REGX src, REGX base, int offset);
rvj_instr rvj_addiw(REGX rd, REGX rs1, int imm);
rvj_instr rvj_slliw(REGX rd, REGX rs1, int shamt);
rvj_instr rvj_srliw(REGX rd, REGX rs1, int shamt);
rvj_instr rvj_sraiw(REGX rd, REGX rs1, int shamt);
rvj_instr rvj_addw(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_subw(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_sllw(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_srlw(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_sraw(REGX rd, REGX rs1, REGX rs2);
/******************************************************************************
** RV32/RV64 Zifencei Standard Extension
******************************************************************************/
rvj_instr rvj_fence_i(REGX rd, REGX rs1, int imm);
/******************************************************************************
** RV32/RV64 Zicsr Standard Extension
******************************************************************************/
rvj_instr rvj_csrrw(REGX rd, REGX rs1, unsigned int csr);
rvj_instr rvj_csrrs(REGX rd, REGX rs1, unsigned int csr);
rvj_instr rvj_csrrc(REGX rd, REGX rs1, unsigned int csr);
rvj_instr rvj_csrrwi(REGX rd, unsigned int uimm, unsigned int csr);
rvj_instr rvj_csrrsi(REGX rd, unsigned int uimm, unsigned int csr);
rvj_instr rvj_csrrci(REGX rd, unsigned int uimm, unsigned int csr);
/******************************************************************************
** RV32-M Standard Extension
******************************************************************************/
rvj_instr rvj_mul(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_mulh(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_mulhsu(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_mulhu(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_div(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_divu(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_rem(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_remu(REGX rd, REGX rs1, REGX rs2);
/******************************************************************************
** RV64-M Standard Extension
******************************************************************************/
rvj_instr rvj_mulw(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_divw(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_divuw(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_remw(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_remuw(REGX rd, REGX rs1, REGX rs2);
/******************************************************************************
** RV32-A Standard Extension (TODO)
******************************************************************************/
/******************************************************************************
** RV64-A Standard Extension (TODO)
******************************************************************************/
/******************************************************************************
** RV32-F Standard Extension
******************************************************************************/
rvj_instr rvj_flw(REGF rd, REGX rs1, int offset);
rvj_instr rvj_fsw(REGX rd, REGF rs1, int offset);
rvj_instr rvj_fmadd_s(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fmsub_s(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fnmsub_s(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fnmadd_s(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fadd_s(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fsub_s(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fmul_s(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fdiv_s(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fsqrt_s(REGF rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fsgnj_s(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fsgnjn_s(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fsgnjx_s(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fmin_s(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fmax_s(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fcvt_w_s(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_wu_s(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fmv_x_w(REGX rd, REGF rs1);
rvj_instr rvj_feq_s(REGX rd, REGF rs1, REGF rs2);
rvj_instr rvj_flt_s(REGX rd, REGF rs1, REGF rs2);
rvj_instr rvj_fle_s(REGX rd, REGF rs1, REGF rs2);
rvj_instr rvj_fclass_s(REGX rd, REGF rs1);
rvj_instr rvj_fcvt_s_w(REGF rd, REGX rs1, rvj_rm rm);
rvj_instr rvj_fcvt_s_wu(REGF rd, REGX rs1, rvj_rm rm);
rvj_instr rvj_fmv_w_x(REGF rd, REGX rs1);
/******************************************************************************
** RV64-F Standard Extension
******************************************************************************/
rvj_instr rvj_fcvt_l_s(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_lu_s(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_s_l(REGF rd, REGX rs1, rvj_rm rm);
rvj_instr rvj_fcvt_s_lu(REGF rd, REGX rs1, rvj_rm rm);
/******************************************************************************
** RV32-D Standard Extension
******************************************************************************/
rvj_instr rvj_fld(REGF rd, REGX rs1, int offset);
rvj_instr rvj_fsd(REGX rd, REGF rs1, int offset);
rvj_instr rvj_fmadd_d(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fmsub_d(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fnmsub_d(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fnmadd_d(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fadd_d(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fsub_d(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fmul_d(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fdiv_d(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fsqrt_d(REGF rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fsgnj_d(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fsgnjn_d(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fsgnjx_d(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fmin_d(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fmax_d(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fcvt_s_d(REGF rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_d_s(REGF rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_feq_d(REGX rd, REGF rs1, REGF rs2);
rvj_instr rvj_flt_d(REGX rd, REGF rs1, REGF rs2);
rvj_instr rvj_fle_d(REGX rd, REGF rs1, REGF rs2);
rvj_instr rvj_fclass_d(REGX rd, REGF rs1);
rvj_instr rvj_fcvt_w_d(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_wu_d(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_d_w(REGF rd, REGX rs1, rvj_rm rm);
rvj_instr rvj_fcvt_d_wu(REGF rd, REGX rs1, rvj_rm rm);
/******************************************************************************
** RV64-D Standard Extension
******************************************************************************/
rvj_instr rvj_fcvt_l_d(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_lu_d(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fmv_x_d(REGX rd, REGF rs1);
rvj_instr rvj_fcvt_d_l(REGF rd, REGX rs1, rvj_rm rm);
rvj_instr rvj_fcvt_d_lu(REGF rd, REGX rs1, rvj_rm rm);
rvj_instr rvj_fmv_d_x(REGF rd, REGX rs1);
/******************************************************************************
** RV32-Q Standard Extension
******************************************************************************/
rvj_instr rvj_flq(REGF rd, REGX rs1, int offset);
rvj_instr rvj_fsq(REGX rd, REGF rs1, int offset);
rvj_instr rvj_fmadd_q(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fmsub_q(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fnmsub_q(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fnmadd_q(REGF rd, REGF rs1, REGF rs2, REGF rs3, rvj_rm rm);
rvj_instr rvj_fadd_q(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fsub_q(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fmul_q(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fdiv_q(REGF rd, REGF rs1, REGF rs2, rvj_rm rm);
rvj_instr rvj_fsqrt_q(REGF rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fsgnj_q(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fsgnjn_q(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fsgnjx_q(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fmin_q(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fmax_q(REGF rd, REGF rs1, REGF rs2);
rvj_instr rvj_fcvt_s_q(REGF rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_q_s(REGF rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_d_q(REGF rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_q_d(REGF rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_feq_q(REGX rd, REGF rs1, REGF rs2);
rvj_instr rvj_flt_q(REGX rd, REGF rs1, REGF rs2);
rvj_instr rvj_fle_q(REGX rd, REGF rs1, REGF rs2);
rvj_instr rvj_fclass_q(REGX rd, REGF rs1);
rvj_instr rvj_fcvt_w_q(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_wu_q(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_q_w(REGF rd, REGX rs1, rvj_rm rm);
rvj_instr rvj_fcvt_q_wu(REGF rd, REGX rs1, rvj_rm rm);
/******************************************************************************
** RV64-Q Standard Extension
******************************************************************************/
rvj_instr rvj_fcvt_l_q(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_lu_q(REGX rd, REGF rs1, rvj_rm rm);
rvj_instr rvj_fcvt_q_l(REGF rd, REGX rs1, rvj_rm rm);
rvj_instr rvj_fcvt_q_lu(REGF rd, REGX rs1, rvj_rm rm);
/******************************************************************************
** RV64-V Standard Extension (TODO: incomplete)
******************************************************************************/
// Vector Control
rvj_instr rvj_vsetvl(REGX rd, REGX rs1, REGX rs2);
rvj_instr rvj_vsetvli(REGX rd, REGX rs1, unsigned int vtypei);
rvj_instr rvj_vsetivli(REGX rd, unsigned int uimm, unsigned int vtypei);
// Vector Load (unit-stride)
rvj_instr rvj_vl(REGV vd, REGX rs1, rvj_vmask vm, unsigned int eew_bytes);
rvj_instr rvj_vle8(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle16(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle32(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle64(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle128(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle256(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle512(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle1024(REGV vd, REGX rs1, rvj_vmask vm);
// Vector Store (unit-stride)
rvj_instr rvj_vs(REGV vs3, REGX rs1, rvj_vmask vm, unsigned int eew_bytes);
rvj_instr rvj_vse8(REGV vs3, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vse16(REGV vs3, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vse32(REGV vs3, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vse64(REGV vs3, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vse128(REGV vs3, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vse256(REGV vs3, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vse512(REGV vs3, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vse1024(REGV vs3, REGX rs1, rvj_vmask vm);
// Vector Load (constant stride)
rvj_instr rvj_vls(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm, unsigned int eew_bytes);
rvj_instr rvj_vlse8(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vlse16(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vlse32(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vlse64(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vlse128(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vlse256(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vlse512(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vlse1024(REGV vd, REGX rs1, REGX rs2, rvj_vmask vm);
// Vector Store (constant stride)
rvj_instr rvj_vss(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm, unsigned int eew_bytes);
rvj_instr rvj_vsse8(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vsse16(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vsse32(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vsse64(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vsse128(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vsse256(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vsse512(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm);
rvj_instr rvj_vsse1024(REGV vs3, REGX rs1, REGX rs2, rvj_vmask vm);
// Vector Load (unordered indexed)
rvj_instr rvj_vlux(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm, unsigned int eew_bytes);
rvj_instr rvj_vluxei8(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vluxei16(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vluxei32(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vluxei64(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vluxei128(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vluxei256(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vluxei512(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vluxei1024(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
// Vector Store (unordered indexed)
rvj_instr rvj_vsux(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm, unsigned int eew_bytes);
rvj_instr rvj_vsuxei8(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsuxei16(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsuxei32(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsuxei64(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsuxei128(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsuxei256(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsuxei512(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsuxei1024(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
// Vector Load (ordered indexed)
rvj_instr rvj_vlox(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm, unsigned int eew_bytes);
rvj_instr rvj_vloxei8(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vloxei16(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vloxei32(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vloxei64(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vloxei128(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vloxei256(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vloxei512(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vloxei1024(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
// Vector Store (ordered indexed)
rvj_instr rvj_vsox(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm, unsigned int eew_bytes);
rvj_instr rvj_vsoxei8(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsoxei16(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsoxei32(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsoxei64(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsoxei128(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsoxei256(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsoxei512(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsoxei1024(REGV vs3, REGX rs1, REGV vs2, rvj_vmask vm);
// Vector Unit-stride Fault-Only-First Loads
rvj_instr rvj_vlff(REGV vd, REGX rs1, rvj_vmask vm, unsigned int eew_bytes);
rvj_instr rvj_vle8ff(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle16ff(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle32ff(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle64ff(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle128ff(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle256ff(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle512ff(REGV vd, REGX rs1, rvj_vmask vm);
rvj_instr rvj_vle1024ff(REGV vd, REGX rs1, rvj_vmask vm);
// Vector Load/Store Segment Instructions (TODO)
// Vector Load/Store Whole Register Instructions (TODO)
// Vector AMO Instructions (TODO)
// Vector Integer Arithmetic Instructions (TODO)

// Vector Integer Add
rvj_instr rvj_vadd_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vadd_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vadd_vi(REGV vd, REGV vs2, int simm5, rvj_vmask vm);
// Vector Integer Subtraction
rvj_instr rvj_vsub_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vsub_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
// Vector Integer Reverse Subtraction
rvj_instr rvj_vrsub_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vrsub_vi(REGV vd, REGV vs2, int simm5, rvj_vmask vm);
// Vector Bitwise AND
rvj_instr rvj_vand_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vand_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vand_vi(REGV vd, REGV vs2, int simm5, rvj_vmask vm);
// Vector Bitwise OR
rvj_instr rvj_vor_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vor_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vor_vi(REGV vd, REGV vs2, int simm5, rvj_vmask vm);
// Vector Bitwise XOR
rvj_instr rvj_vxor_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vxor_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vxor_vi(REGV vd, REGV vs2, int simm5, rvj_vmask vm);
// Vector Integer Divide
rvj_instr rvj_vdiv_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vdiv_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
// Vector Integer Multiply
rvj_instr rvj_vmul_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vmul_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
// Vector Integer Fused Multiply and Add (vd is first multiplicand)
rvj_instr rvj_vmadd_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vmadd_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
// Vector Integer Fused Multiply and Add (vd is addend)
rvj_instr rvj_vmacc_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vmacc_vx(REGV vd, REGX rs1, REGV vs2, rvj_vmask vm);
// Vector Integer Merge
rvj_instr rvj_vmerge_vvm(REGV vd, REGV vs2, REGV vs1);
rvj_instr rvj_vmerge_vxm(REGV vd, REGV vs2, REGX rs1);
rvj_instr rvj_vmerge_vim(REGV vd, REGV vs2, int simm5);
// Vector Integer Move
rvj_instr rvj_vmv_vv(REGV vd, REGV vs1);
rvj_instr rvj_vmv_vx(REGV vd, REGX rs1);
rvj_instr rvj_vmv_vi(REGV vd, int simm5);

// Vector Fixed-Point Arithmetic Instructions (TODO)

// Vector Floating-Point Arithmetic Operations (TODO)
// Vector Floating-point Add
rvj_instr rvj_vfadd_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vfadd_vf(REGV vd, REGF rs1, REGV vs2, rvj_vmask vm);
// Vector Floating-point Subtraction
rvj_instr rvj_vfsub_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vfsub_vf(REGV vd, REGF rs1, REGV vs2, rvj_vmask vm);
// Vector Floating-point Reverse Subtraction
rvj_instr rvj_vfrsub_vf(REGV vd, REGF rs1, REGV vs2, rvj_vmask vm);
// Vector Floating-point Divide
rvj_instr rvj_vfdiv_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vfdiv_vf(REGV vd, REGF rs1, REGV vs2, rvj_vmask vm);
// Vector Floating-point Reverse Divide
rvj_instr rvj_vfrdiv_vf(REGV vd, REGF rs1, REGV vs2, rvj_vmask vm);
// Vector Floating-point Multiply
rvj_instr rvj_vfmul_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vfmul_vf(REGV vd, REGF rs1, REGV vs2, rvj_vmask vm);
// Vector Floating-point Fused Multiply and Add (vd is first multiplicand)
rvj_instr rvj_vfmadd_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vfmadd_vf(REGV vd, REGF rs1, REGV vs2, rvj_vmask vm);
// Vector Floating-point Fused Multiply and Add (vd is addend)
rvj_instr rvj_vfmacc_vv(REGV vd, REGV vs1, REGV vs2, rvj_vmask vm);
rvj_instr rvj_vfmacc_vf(REGV vd, REGF rs1, REGV vs2, rvj_vmask vm);

// Vector Reduction Operations (TODO)
// Vector Mask Instructions (TODO)
rvj_instr rvj_vmand_vv(REGV vd, REGV vs1, REGV vs2);
rvj_instr rvj_vmslt_vx(REGV vd, REGV vs1, REGV vs2);
rvj_instr rvj_vmseq_vv(REGV vd, REGV vs1, REGV vs2);
rvj_instr rvj_vmsgt_vx(REGV vd, REGV vs1, REGV vs2);
// Vector Permutation Instructions (TODO)
// Vector Integer Scalar Move Instructions
rvj_instr rvj_vmv_xs(REGX vd, REGV vs2);
rvj_instr rvj_vmv_sx(REGV vd, REGX rs1);
// Vector Floating-Point Scalar Move Instructions
rvj_instr rvj_vfmv_fs(REGF vd, REGV vs2);
rvj_instr rvj_vfmv_sf(REGV vd, REGF rs1);
// ... (TODO) ...
// Whole Vector Register Move
rvj_instr rvj_vmvr_v(REGV vd, REGV vs2, unsigned int nr);
rvj_instr rvj_vmv1r_v(REGV vd, REGV vs2);
rvj_instr rvj_vmv2r_v(REGV vd, REGV vs2);
rvj_instr rvj_vmv4r_v(REGV vd, REGV vs2);
rvj_instr rvj_vmv8r_v(REGV vd, REGV vs2);


/******************************************************************************
** RV64-V Standard Extension Pseudo Instructions
******************************************************************************/
rvj_instr rvj_vsetvlmax(REGX rd, REGX rs2);
rvj_instr rvj_vsetvlmaxi(REGX rd, unsigned int vtypei);

#undef REGX
#undef REGF
#undef REGV

#ifdef __cplusplus
}
#endif

#endif