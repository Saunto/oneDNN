#pragma once

#include <cstdlib>
#include <type_traits>
#include "rvjit.h"

namespace rvjit {

/// @brief RISC-V JIT return status code
using status_t = rvj_status_t;

/// @brief RISC-V binary instruction
using instr_t = rvj_instr;

/// @brief Type that represents a RISC-V named general purpose register
using gpr_t = rvj_gpr;

/// @brief Type that represents a RISC-V named floating-point register
using fpr_t = rvj_fpr;

/// @brief Type that represents a RISC-V named vector register
using vr_t = rvj_vr;

/// @brief Rounding modes supported on floating-points instructions
using rm_t = rvj_rm;

/// @brief Bitmask for composing the contents of the vtype field for the
/// V standard extension vsetvl instructions
using vtype_t = rvj_vtype_mask;

/// @brief Flag to determines if vector instructions use a mask register
using vmask_t = rvj_vmask;

#define define_status(s) const status_t s = rvj_##s
namespace status {
    define_status(success);           /// Success
    define_status(invalid_arguments); /// Invalid user inputs
    define_status(empty);             /// Code Region is empty
    define_status(out_of_memory);     /// Failure to allocate memory
    define_status(error);             /// Internal error (bug)
    define_status(undefined_label);   /// Label used but not defined
};
#undef define_status

namespace rounding_mode {
    const rm_t rne = rvj_rne; // Round to Nearest, ties to Even
    const rm_t rtz = rvj_rtz; // Round towards Zero
    const rm_t rdn = rvj_rdn; // Round Down (towards − infinity)
    const rm_t rup = rvj_rup; // Round Up (towards + infinity)
    const rm_t rmm = rvj_rmm; // Round to Nearest, ties to Max Magnitude
    const rm_t dyn = rvj_dyn; // Selects dynamic rounding mode
    const rm_t default_rm = rne; // The library default rounding mode
};

#define define_vtype(v) const vtype_t v = rvj_##v
namespace vtype {
#ifdef RVJ_VSPEC_1_0
    define_vtype(e8);    // Set single element width size to 8 bytes
    define_vtype(e16);   // Set single element width size to 16 bytes
    define_vtype(e32);   // Set single element width size to 32 bytes
    define_vtype(e64);   // Set single element width size to 64 bytes
    define_vtype(e128);  // Set single element width size to 128 bytes
    define_vtype(e256);  // Reserved but unsupported officially
    define_vtype(e512);  // Reserved but unsupported officially
    define_vtype(e1024); // Reserved but unsupported officially
    define_vtype(vma);   // Set mask-policy to 'vector mask agnostic'
    define_vtype(vta);   // Set tail-policy to 'vector tail agnostic'
    define_vtype(m1);    // Set the length multiplier to 1
    define_vtype(m2);    // Set the length multiplier to 2
    define_vtype(m4);    // Set the length multiplier to 4
    define_vtype(m8);    // Set the length multiplier to 8
    define_vtype(mf8);   // Set the length multiplier to 1/8
    define_vtype(mf4);   // Set the length multiplier to 1/4
    define_vtype(mf2);   // Set the length multiplier to 1/2
#else
    define_vtype(e8);    // Set single element width size to 8 bytes
    define_vtype(e16);   // Set single element width size to 16 bytes
    define_vtype(e32);   // Set single element width size to 32 bytes
    define_vtype(e64);   // Set single element width size to 64 bytes
    define_vtype(e128);  // Set single element width size to 128 bytes
    define_vtype(e256);  // Reserved but unsupported officially
    define_vtype(e512);  // Reserved but unsupported officially
    define_vtype(e1024); // Reserved but unsupported officially
    define_vtype(m1);    // Set the length multiplier to 1
    define_vtype(m2);    // Set the length multiplier to 2
    define_vtype(m4);    // Set the length multiplier to 4
    define_vtype(m8);    // Set the length multiplier to 8
#endif
};
#undef define_vtype

namespace vmask {
    const vmask_t unmasked = rvj_unmasked;
    const vmask_t v0_t = rvj_masked;
};

// TODO: Wrappers for errors that throw std::runtime()

class function {
private:
    rvj_function_t handle;
    void *ptr;

public:
    function() : handle(nullptr), ptr(nullptr) {}

    function(rvj_asm_t assembler) : handle(nullptr), ptr(nullptr) {
        rvj_asm_get_function_handle(assembler, &handle);
        rvj_function_get_pointer(handle, &ptr);
    }

    virtual ~function() {
        if (handle) {
            rvj_function_free(&handle);
            handle = nullptr;
            ptr = nullptr;
        }
    }

    size_t size() const {
        return handle ? rvj_function_get_size(handle) : 0;
    }

    size_t dump(char * const buf) const {
        return handle ? rvj_function_dump(handle, buf) : 0;
    }

    template<typename T = void(*)(), typename B = typename std::enable_if<
        std::is_pointer<T>::value &&
        std::is_function<typename std::remove_pointer<T>::type>::value>::type>
    T get() const {
        return reinterpret_cast<T>(ptr);
    }
};

class assembler {
private:
    rvj_asm_t handle;
    function *f;
    bool is_done;

    status_t push(const instr_t i) { return rvj_asm_push(handle, i); }

    status_t push_lref(const instr_t i, const char *label) {
        return rvj_asm_push_lref(handle, i, label);
    }

public:
    assembler() : handle(nullptr), f(nullptr), is_done(false) {
        rvj_asm_init(&handle);
    }
 
    virtual ~assembler() { rvj_asm_free(&handle); }

    function* assemble() {
        if (is_done)
            return f;
        status_t s = rvj_asm_done(handle);
        if (s != status::success)
            return nullptr;
        f = new function(handle);
        is_done = true;
        return f;
    }

    function* get_function() { return f; }

 protected:
    const gpr_t x0 = rvj_x0;
    const gpr_t ra = rvj_ra;
    const gpr_t sp = rvj_sp;
    const gpr_t gp = rvj_gp;
    const gpr_t tp = rvj_tp;
    const gpr_t t0 = rvj_t0;
    const gpr_t t1 = rvj_t1;
    const gpr_t t2 = rvj_t2;
    const gpr_t s0 = rvj_s0;
    const gpr_t s1 = rvj_s1;
    const gpr_t a0 = rvj_a0;
    const gpr_t a1 = rvj_a1;
    const gpr_t a2 = rvj_a2;
    const gpr_t a3 = rvj_a3;
    const gpr_t a4 = rvj_a4;
    const gpr_t a5 = rvj_a5;
    const gpr_t a6 = rvj_a6;
    const gpr_t a7 = rvj_a7;
    const gpr_t s2 = rvj_s2;
    const gpr_t s3 = rvj_s3;
    const gpr_t s4 = rvj_s4;
    const gpr_t s5 = rvj_s5;
    const gpr_t s6 = rvj_s6;
    const gpr_t s7 = rvj_s7;
    const gpr_t s8 = rvj_s8;
    const gpr_t s9 = rvj_s9;
    const gpr_t s10 = rvj_s10;
    const gpr_t s11 = rvj_s11;
    const gpr_t t3 = rvj_t3;
    const gpr_t t4 = rvj_t4;
    const gpr_t t5 = rvj_t5;
    const gpr_t t6 = rvj_t6;
    const fpr_t ft0 = rvj_ft0;
    const fpr_t ft1 = rvj_ft1;
    const fpr_t ft2 = rvj_ft2;
    const fpr_t ft3 = rvj_ft3;
    const fpr_t ft4 = rvj_ft4;
    const fpr_t ft5 = rvj_ft5;
    const fpr_t ft6 = rvj_ft6;
    const fpr_t ft7 = rvj_ft7;
    const fpr_t fs0 = rvj_fs0;
    const fpr_t fs1 = rvj_fs1;
    const fpr_t fa0 = rvj_fa0;
    const fpr_t fa1 = rvj_fa1;
    const fpr_t fa2 = rvj_fa2;
    const fpr_t fa3 = rvj_fa3;
    const fpr_t fa4 = rvj_fa4;
    const fpr_t fa5 = rvj_fa5;
    const fpr_t fa6 = rvj_fa6;
    const fpr_t fa7 = rvj_fa7;
    const fpr_t fs2 = rvj_fs2;
    const fpr_t fs3 = rvj_fs3;
    const fpr_t fs4 = rvj_fs4;
    const fpr_t fs5 = rvj_fs5;
    const fpr_t fs6 = rvj_fs6;
    const fpr_t fs7 = rvj_fs7;
    const fpr_t fs8 = rvj_fs8;
    const fpr_t fs9 = rvj_fs9;
    const fpr_t fs10 = rvj_fs10;
    const fpr_t fs11 = rvj_fs11;
    const fpr_t ft8 = rvj_ft8;
    const fpr_t ft9 = rvj_ft9;
    const fpr_t ft10 = rvj_ft10;
    const fpr_t ft11 = rvj_ft11;
    const vr_t v0 = rvj_v0;
    const vr_t v1 = rvj_v1;
    const vr_t v2 = rvj_v2;
    const vr_t v3 = rvj_v3;
    const vr_t v4 = rvj_v4;
    const vr_t v5 = rvj_v5;
    const vr_t v6 = rvj_v6;
    const vr_t v7 = rvj_v7;
    const vr_t v8 = rvj_v8;
    const vr_t v9 = rvj_v9;
    const vr_t v10 = rvj_v10;
    const vr_t v11 = rvj_v11;
    const vr_t v12 = rvj_v12;
    const vr_t v13 = rvj_v13;
    const vr_t v14 = rvj_v14;
    const vr_t v15 = rvj_v15;
    const vr_t v16 = rvj_v16;
    const vr_t v17 = rvj_v17;
    const vr_t v18 = rvj_v18;
    const vr_t v19 = rvj_v19;
    const vr_t v20 = rvj_v20;
    const vr_t v21 = rvj_v21;
    const vr_t v22 = rvj_v22;
    const vr_t v23 = rvj_v23;
    const vr_t v24 = rvj_v24;
    const vr_t v25 = rvj_v25;
    const vr_t v26 = rvj_v26;
    const vr_t v27 = rvj_v27;
    const vr_t v28 = rvj_v28;
    const vr_t v29 = rvj_v29;
    const vr_t v30 = rvj_v30;
    const vr_t v31 = rvj_v31;

    gpr_t get_gpr(const char *name) const {
        gpr_t o;
        rvj_get_gpr_id(name, &o);
        return o;
    }

    fpr_t get_fpr(const char *name) const {
        fpr_t o;
        rvj_get_fpr_id(name, &o);
        return o;
    }

    vr_t get_vr(const char *name) const {
        vr_t o;
        rvj_get_vr_id(name, &o);
        return o;
    }

    void L(const char* name) {
        rvj_asm_label(handle, name);
    }

protected:
    vtype_t vsew(int size) const {
        switch (size) {
            case 2: return vtype::e16;
            case 4: return vtype::e32;
            case 8: return vtype::e64;
            case 16: return vtype::e128;
            case 32: return vtype::e256;
            case 64: return vtype::e512;
            case 128: return vtype::e1024;
            default: return vtype::e8;
        }
    }

    vtype_t vlmul(int l, bool frac = false) const {
        #ifdef RVJ_VSPEC_1_0
        switch(l) {
            case 2:  return frac ? vtype::mf2 : vtype::m2;
            case 4:  return frac ? vtype::mf4 : vtype::m4;
            case 8:  return frac ? vtype::mf8 : vtype::m8;
            default: return vtype::m1;
        }
        #else
        switch(l) {
            case 2:  return vtype::m2;
            case 4:  return vtype::m4;
            case 8:  return vtype::m8;
            default: return vtype::m1;
        }
        #endif
    }

    /**************************************************************************
    ** RV32-I Base Instruction Set
    **************************************************************************/
    void lui(gpr_t rd, int imm) {
        push(rvj_lui(rd, imm));
    }
    
    void auipc(gpr_t rd, int imm) {
        push(rvj_auipc(rd, imm));
    }
    
    void jal(gpr_t rd, int offset) {
        push(rvj_jal(rd, offset));
    }
    
    void jalr(gpr_t rd, gpr_t rs1, int offset) {
        push(rvj_jalr(rd, rs1, offset));
    }
    
    void beq(gpr_t rs1, gpr_t rs2, int offset) {
        push(rvj_beq(rs1, rs2, offset));
    }
    
    void bne(gpr_t rs1, gpr_t rs2, int offset) {
        push(rvj_bne(rs1, rs2, offset));
    }
    
    void blt(gpr_t rs1, gpr_t rs2, int offset) {
        push(rvj_blt(rs1, rs2, offset));
    }
    
    void bge(gpr_t rs1, gpr_t rs2, int offset) {
        push(rvj_bge(rs1, rs2, offset));
    }
    
    void bltu(gpr_t rs1, gpr_t rs2, int offset) {
        push(rvj_bltu(rs1, rs2, offset));
    }
    
    void bgeu(gpr_t rs1, gpr_t rs2, int offset) {
        push(rvj_bgeu(rs1, rs2, offset));
    }
    
    void lb(gpr_t rd, gpr_t rs1, int offset) {
        push(rvj_lb(rd, rs1, offset));
    }
    
    void lh(gpr_t rd, gpr_t rs1, int offset) {
        push(rvj_lh(rd, rs1, offset));
    }
    
    void lw(gpr_t rd, gpr_t rs1, int offset) {
        push(rvj_lw(rd, rs1, offset));
    }
    
    void lbu(gpr_t rd, gpr_t rs1, int offset) {
        push(rvj_lbu(rd, rs1, offset));
    }
    
    void lhu(gpr_t rd, gpr_t rs1, int offset) {
        push(rvj_lhu(rd, rs1, offset));
    }
    
    void sb(gpr_t src, gpr_t base, int offset) {
        push(rvj_sb(src, base, offset));
    }
    
    void sh(gpr_t src, gpr_t base, int offset) {
        push(rvj_sh(src, base, offset));
    }
    
    void sw(gpr_t src, gpr_t base, int offset) {
        push(rvj_sw(src, base, offset));
    }
    
    void addi(gpr_t rd, gpr_t rs1, int imm) {
        push(rvj_addi(rd, rs1, imm));
    }
    
    void slti(gpr_t rd, gpr_t rs1, int imm) {
        push(rvj_slti(rd, rs1, imm));
    }
    
    void sltiu(gpr_t rd, gpr_t rs1, int imm) {
        push(rvj_sltiu(rd, rs1, imm));
    }
    
    void xori(gpr_t rd, gpr_t rs1, int imm) {
        push(rvj_xori(rd, rs1, imm));
    }
    
    void ori(gpr_t rd, gpr_t rs1, int imm) {
        push(rvj_ori(rd, rs1, imm));
    }
    
    void andi(gpr_t rd, gpr_t rs1, int imm) {
        push(rvj_andi(rd, rs1, imm));
    }
    
    void slli(gpr_t rd, gpr_t rs1, int shamt) {
        push(rvj_slli(rd, rs1, shamt));
    }
    
    void srli(gpr_t rd, gpr_t rs1, int shamt) {
        push(rvj_srli(rd, rs1, shamt));
    }
    
    void srai(gpr_t rd, gpr_t rs1, int shamt) {
        push(rvj_srai(rd, rs1, shamt));
    }
    
    void add(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_add(rd, rs1, rs2));
    }
    
    void sub(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_sub(rd, rs1, rs2));
    }
    
    void sll(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_sll(rd, rs1, rs2));
    }
    
    void slt(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_slt(rd, rs1, rs2));
    }
    
    void sltu(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_sltu(rd, rs1, rs2));
    }
    
    void xor_(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_xor_(rd, rs1, rs2));
    }
    
    void srl(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_srl(rd, rs1, rs2));
    }
    
    void sra(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_sra(rd, rs1, rs2));
    }
    
    void or_(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_or_(rd, rs1, rs2));
    }
    
    void and_(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_and_(rd, rs1, rs2));
    }
    
    void fence(gpr_t rd, gpr_t rs1, int imm) {
        push(rvj_fence(rd, rs1, imm));
    }
    
    void ecall() {
        push(rvj_ecall());
    }
    
    void ebreak() {
        push(rvj_ebreak());
    }
    
    /**************************************************************************
    ** RV32-I Pseudo Instructions
    **************************************************************************/
    void nop() {
        push(rvj_nop());
    }
    
    void ret() {
        push(rvj_ret());
    }
    
    void mv(gpr_t rd, gpr_t rs1) {
        push(rvj_mv(rd, rs1));
    }
    
    void li(gpr_t rd, const int imm) {
        push(rvj_li(rd, imm));
    }
    
    void j(const int imm) {
        push(rvj_j(imm));
    }
    
    void beqz(gpr_t rd, int offset) {
        push(rvj_beqz(rd, offset));
    }
    
    void bnez(gpr_t rd, int offset) {
        push(rvj_bnez(rd, offset));
    }
    
    void sext_w(gpr_t rd, gpr_t rs1) {
        push(rvj_sext_w(rd, rs1));
    }
    
    /**************************************************************************
    ** Pseudo Instructions Instructions Using Labels For Offsets
    **************************************************************************/
    void j(const char *label) {
        push_lref(rvj_j(0), label);
    }
    
    void beqz(gpr_t rd, const char *label) {
        push_lref(rvj_beqz(rd, 0), label);
    }
    
    void bnez(gpr_t rd, const char *label) {
        push_lref(rvj_bnez(rd, 0), label);
    }
    
    /**************************************************************************
    ** RV32-I Control Transfer Instructions Using Labels For Offsets
    **************************************************************************/
    void jal(gpr_t rd, const char *label) {
        push_lref(rvj_jal(rd, 0), label);
    }
    
    void beq(gpr_t rs1, gpr_t rs2, const char *label) {
        push_lref(rvj_beq(rs1, rs2, 0), label);
    }
    
    void bne(gpr_t rs1, gpr_t rs2, const char *label) {
        push_lref(rvj_bne(rs1, rs2, 0), label);
    }
    
    void blt(gpr_t rs1, gpr_t rs2, const char *label) {
        push_lref(rvj_blt(rs1, rs2, 0), label);
    }
    
    void bge(gpr_t rs1, gpr_t rs2, const char *label) {
        push_lref(rvj_bge(rs1, rs2, 0), label);
    }
    
    void bltu(gpr_t rs1, gpr_t rs2, const char *label) {
        push_lref(rvj_bltu(rs1, rs2, 0), label);
    }
    
    void bgeu(gpr_t rs1, gpr_t rs2, const char *label) {
        push_lref(rvj_bgeu(rs1, rs2, 0), label);
    }
    
    /**************************************************************************
    ** RV64-I Standard Extension
    **************************************************************************/
    void lwu(gpr_t rd, gpr_t rs1, int offset) {
        push(rvj_lwu(rd, rs1, offset));
    }
    
    void ld(gpr_t rd, gpr_t rs1, int offset) {
        push(rvj_ld(rd, rs1, offset));
    }
    
    void sd(gpr_t src, gpr_t base, int offset) {
        push(rvj_sd(src, base, offset));
    }
    
    void addiw(gpr_t rd, gpr_t rs1, int imm) {
        push(rvj_addiw(rd, rs1, imm));
    }
    
    void slliw(gpr_t rd, gpr_t rs1, int shamt) {
        push(rvj_slliw(rd, rs1, shamt));
    }
    
    void srliw(gpr_t rd, gpr_t rs1, int shamt) {
        push(rvj_srliw(rd, rs1, shamt));
    }
    
    void sraiw(gpr_t rd, gpr_t rs1, int shamt) {
        push(rvj_sraiw(rd, rs1, shamt));
    }
    
    void addw(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_addw(rd, rs1, rs2));
    }
    
    void subw(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_subw(rd, rs1, rs2));
    }
    
    void sllw(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_sllw(rd, rs1, rs2));
    }
    
    void srlw(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_srlw(rd, rs1, rs2));
    }
    
    void sraw(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_sraw(rd, rs1, rs2));
    }
    
    /**************************************************************************
    ** RV32/RV64 Zifencei Standard Extension
    **************************************************************************/
    void fence_i(gpr_t rd, gpr_t rs1, int imm) {
        push(rvj_fence_i(rd, rs1, imm));
    }
    
    /**************************************************************************
    ** RV32/RV64 Zicsr Standard Extension
    **************************************************************************/
    void csrrw(gpr_t rd, gpr_t rs1, unsigned int csr) {
        push(rvj_csrrw(rd, rs1, csr));
    }
    
    void csrrs(gpr_t rd, gpr_t rs1, unsigned int csr) {
        push(rvj_csrrs(rd, rs1, csr));
    }
    
    void csrrc(gpr_t rd, gpr_t rs1, unsigned int csr) {
        push(rvj_csrrc(rd, rs1, csr));
    }
    
    void csrrwi(gpr_t rd, unsigned int uimm, unsigned int csr) {
        push(rvj_csrrwi(rd, uimm, csr));
    }
    
    void csrrsi(gpr_t rd, unsigned int uimm, unsigned int csr) {
        push(rvj_csrrsi(rd, uimm, csr));
    }
    
    void csrrci(gpr_t rd, unsigned int uimm, unsigned int csr) {
        push(rvj_csrrci(rd, uimm, csr));
    }
    
    /**************************************************************************
    ** RV32-M Standard Extension
    **************************************************************************/
    void mul(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_mul(rd, rs1, rs2));
    }
    
    void mulh(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_mulh(rd, rs1, rs2));
    }
    
    void mulhsu(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_mulhsu(rd, rs1, rs2));
    }
    
    void mulhu(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_mulhu(rd, rs1, rs2));
    }
    
    void div(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_div(rd, rs1, rs2));
    }
    
    void divu(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_divu(rd, rs1, rs2));
    }
    
    void rem(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_rem(rd, rs1, rs2));
    }
    
    void remu(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_remu(rd, rs1, rs2));
    }
    
    /**************************************************************************
    ** RV64-M Standard Extension
    **************************************************************************/
    void mulw(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_mulw(rd, rs1, rs2));
    }
    
    void divw(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_divw(rd, rs1, rs2));
    }
    
    void divuw(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_divuw(rd, rs1, rs2));
    }
    
    void remw(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_remw(rd, rs1, rs2));
    }
    
    void remuw(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_remuw(rd, rs1, rs2));
    }
    
    /**************************************************************************
    ** RV32-A Standard Extension (TODO)
    **************************************************************************/
    /**************************************************************************
    ** RV64-A Standard Extension (TODO)
    **************************************************************************/
    /**************************************************************************
    ** RV32-F Standard Extension
    **************************************************************************/
    void flw(fpr_t rd, gpr_t rs1, int offset) {
        push(rvj_flw(rd, rs1, offset));
    }

    void fsw(gpr_t rd, fpr_t rs1, int offset) {
        push(rvj_fsw(rd, rs1, offset));
    }

    void fmadd_s(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fmadd_s(rd, rs1, rs2, rs3, rm));
    }

    void fmsub_s(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fmsub_s(rd, rs1, rs2, rs3, rm));
    }

    void fnmsub_s(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fnmsub_s(rd, rs1, rs2, rs3, rm));
    }

    void fnmadd_s(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fnmadd_s(rd, rs1, rs2, rs3, rm));
    }

    void fadd_s(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fadd_s(rd, rs1, rs2, rm));
    }

    void fsub_s(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fsub_s(rd, rs1, rs2, rm));
    }

    void fmul_s(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fmul_s(rd, rs1, rs2, rm));
    }

    void fdiv_s(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fdiv_s(rd, rs1, rs2, rm));
    }

    void fsqrt_s(fpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fsqrt_s(rd, rs1, rm));
    }

    void fsgnj_s(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fsgnj_s(rd, rs1, rs2));
    }

    void fsgnjn_s(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fsgnjn_s(rd, rs1, rs2));
    }

    void fsgnjx_s(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fsgnjx_s(rd, rs1, rs2));
    }

    void fmin_s(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fmin_s(rd, rs1, rs2));
    }

    void fmax_s(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fmax_s(rd, rs1, rs2));
    }

    void fcvt_w_s(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_w_s(rd, rs1, rm));
    }

    void fcvt_wu_s(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_wu_s(rd, rs1, rm));
    }

    void fmv_x_w(gpr_t rd, fpr_t rs1) {
        push(rvj_fmv_x_w(rd, rs1));
    }

    void feq_s(gpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_feq_s(rd, rs1, rs2));
    }

    void flt_s(gpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_flt_s(rd, rs1, rs2));
    }

    void fle_s(gpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fle_s(rd, rs1, rs2));
    }

    void fclass_s(gpr_t rd, fpr_t rs1) {
        push(rvj_fclass_s(rd, rs1));
    }

    void fcvt_s_w(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_s_w(rd, rs1, rm));
    }

    void fcvt_s_wu(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_s_wu(rd, rs1, rm));
    }

    void fmv_w_x(fpr_t rd, gpr_t rs1) {
        push(rvj_fmv_w_x(rd, rs1));
    }

    /**************************************************************************
    ** RV64-F Standard Extension
    **************************************************************************/
    void fcvt_l_s(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_l_s(rd, rs1, rm));
    }

    void fcvt_lu_s(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_lu_s(rd, rs1, rm));
    }

    void fcvt_s_l(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_s_l(rd, rs1, rm));
    }

    void fcvt_s_lu(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_s_lu(rd, rs1, rm));
    }

    /**************************************************************************
    ** RV32-D Standard Extension
    **************************************************************************/
    void fld(fpr_t rd, gpr_t rs1, int offset) {
        push(rvj_fld(rd, rs1, offset));
    }

    void fsd(gpr_t rd, fpr_t rs1, int offset) {
        push(rvj_fsd(rd, rs1, offset));
    }

    void fmadd_d(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fmadd_d(rd, rs1, rs2, rs3, rm));
    }

    void fmsub_d(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fmsub_d(rd, rs1, rs2, rs3, rm));
    }

    void fnmsub_d(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fnmsub_d(rd, rs1, rs2, rs3, rm));
    }

    void fnmadd_d(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fnmadd_d(rd, rs1, rs2, rs3, rm));
    }

    void fadd_d(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fadd_d(rd, rs1, rs2, rm));
    }

    void fsub_d(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fsub_d(rd, rs1, rs2, rm));
    }

    void fmul_d(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fmul_d(rd, rs1, rs2, rm));
    }

    void fdiv_d(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fdiv_d(rd, rs1, rs2, rm));
    }

    void fsqrt_d(fpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fsqrt_d(rd, rs1, rm));
    }

    void fsgnj_d(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fsgnj_d(rd, rs1, rs2));
    }

    void fsgnjn_d(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fsgnjn_d(rd, rs1, rs2));
    }

    void fsgnjx_d(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fsgnjx_d(rd, rs1, rs2));
    }

    void fmin_d(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fmin_d(rd, rs1, rs2));
    }

    void fmax_d(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fmax_d(rd, rs1, rs2));
    }

    void fcvt_s_d(fpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_s_d(rd, rs1, rm));
    }

    void fcvt_d_s(fpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_d_s(rd, rs1, rm));
    }

    void feq_d(gpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_feq_d(rd, rs1, rs2));
    }

    void flt_d(gpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_flt_d(rd, rs1, rs2));
    }

    void fle_d(gpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fle_d(rd, rs1, rs2));
    }

    void fclass_d(gpr_t rd, fpr_t rs1) {
        push(rvj_fclass_d(rd, rs1));
    }

    void fcvt_w_d(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_w_d(rd, rs1, rm));
    }

    void fcvt_wu_d(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_wu_d(rd, rs1, rm));
    }

    void fcvt_d_w(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_d_w(rd, rs1, rm));
    }

    void fcvt_d_wu(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_d_wu(rd, rs1, rm));
    }

    /**************************************************************************
    ** RV64-D Standard Extension
    **************************************************************************/
    void fcvt_l_d(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_l_d(rd, rs1, rm));
    }

    void fcvt_lu_d(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_lu_d(rd, rs1, rm));
    }

    void fmv_x_d(gpr_t rd, fpr_t rs1) {
        push(rvj_fmv_x_d(rd, rs1));
    }

    void fcvt_d_l(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_d_l(rd, rs1, rm));
    }

    void fcvt_d_lu(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_d_lu(rd, rs1, rm));
    }

    void fmv_d_x(fpr_t rd, gpr_t rs1) {
        push(rvj_fmv_d_x(rd, rs1));
    }

    /**************************************************************************
    ** RV32-Q Standard Extension
    **************************************************************************/
    void flq(fpr_t rd, gpr_t rs1, int offset) {
        push(rvj_flq(rd, rs1, offset));
    }

    void fsq(gpr_t rd, fpr_t rs1, int offset) {
        push(rvj_fsq(rd, rs1, offset));
    }

    void fmadd_q(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fmadd_q(rd, rs1, rs2, rs3, rm));
    }

    void fmsub_q(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fmsub_q(rd, rs1, rs2, rs3, rm));
    }

    void fnmsub_q(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fnmsub_q(rd, rs1, rs2, rs3, rm));
    }

    void fnmadd_q(fpr_t rd, fpr_t rs1, fpr_t rs2, fpr_t rs3, rvj_rm rm) {
        push(rvj_fnmadd_q(rd, rs1, rs2, rs3, rm));
    }

    void fadd_q(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fadd_q(rd, rs1, rs2, rm));
    }

    void fsub_q(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fsub_q(rd, rs1, rs2, rm));
    }

    void fmul_q(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fmul_q(rd, rs1, rs2, rm));
    }

    void fdiv_q(fpr_t rd, fpr_t rs1, fpr_t rs2, rvj_rm rm) {
        push(rvj_fdiv_q(rd, rs1, rs2, rm));
    }

    void fsqrt_q(fpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fsqrt_q(rd, rs1, rm));
    }

    void fsgnj_q(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fsgnj_q(rd, rs1, rs2));
    }

    void fsgnjn_q(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fsgnjn_q(rd, rs1, rs2));
    }

    void fsgnjx_q(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fsgnjx_q(rd, rs1, rs2));
    }

    void fmin_q(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fmin_q(rd, rs1, rs2));
    }

    void fmax_q(fpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fmax_q(rd, rs1, rs2));
    }

    void fcvt_s_q(fpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_s_q(rd, rs1, rm));
    }

    void fcvt_q_s(fpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_q_s(rd, rs1, rm));
    }

    void fcvt_d_q(fpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_d_q(rd, rs1, rm));
    }

    void fcvt_q_d(fpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_q_d(rd, rs1, rm));
    }

    void feq_q(gpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_feq_q(rd, rs1, rs2));
    }

    void flt_q(gpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_flt_q(rd, rs1, rs2));
    }

    void fle_q(gpr_t rd, fpr_t rs1, fpr_t rs2) {
        push(rvj_fle_q(rd, rs1, rs2));
    }

    void fclass_q(gpr_t rd, fpr_t rs1) {
        push(rvj_fclass_q(rd, rs1));
    }

    void fcvt_w_q(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_w_q(rd, rs1, rm));
    }

    void fcvt_wu_q(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_wu_q(rd, rs1, rm));
    }

    void fcvt_q_w(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_q_w(rd, rs1, rm));
    }

    void fcvt_q_wu(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_q_wu(rd, rs1, rm));
    }

    /**************************************************************************
    ** RV64-Q Standard Extension
    **************************************************************************/
    void fcvt_l_q(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_l_q(rd, rs1, rm));
    }

    void fcvt_lu_q(gpr_t rd, fpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_lu_q(rd, rs1, rm));
    }

    void fcvt_q_l(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_q_l(rd, rs1, rm));
    }

    void fcvt_q_lu(fpr_t rd, gpr_t rs1, rvj_rm rm) {
        push(rvj_fcvt_q_lu(rd, rs1, rm));
    }

    /**************************************************************************
    ** RV64-V Standard Extension (TODO)
    **************************************************************************/
    // Vector Control
    void vsetvl(gpr_t rd, gpr_t rs1, gpr_t rs2) {
        push(rvj_vsetvl(rd, rs1, rs2));
    }

    void vsetvli(gpr_t rd, gpr_t rs1, unsigned int vtypei) {
        push(rvj_vsetvli(rd, rs1, vtypei));
    }

    void vsetivli(gpr_t rd, unsigned int uimm, unsigned int vtypei) {
        push(rvj_vsetivli(rd, uimm, vtypei));
    }

    // Vector Load (unit-stride)
    template<typename T>
    void vl(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vl(vd, rs1, vm, sizeof(T)));
    }

    void vl(vr_t vd, gpr_t rs1, unsigned int sew_bytes,
                                            vmask_t vm = vmask::unmasked) {
        push(rvj_vl(vd, rs1, vm, sew_bytes));
    }

    void vle8(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle8(vd, rs1, vm));
    }

    void vle16(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle16(vd, rs1, vm));
    }

    void vle32(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle32(vd, rs1, vm));
    }

    void vle64(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle64(vd, rs1, vm));
    }

    void vle128(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle128(vd, rs1, vm));
    }

    void vle256(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle256(vd, rs1, vm));
    }

    void vle512(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle512(vd, rs1, vm));
    }

    void vle1024(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle1024(vd, rs1, vm));
    }

    // Vector Store (unit-stride)
    template<typename T>
    void vs(vr_t vs3, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vs(vs3, rs1, vm, sizeof(T)));
    }

    void vs(vr_t vs3, gpr_t rs1, unsigned int sew_bytes,
                                            vmask_t vm = vmask::unmasked) {
        push(rvj_vs(vs3, rs1, vm, sew_bytes));
    }

    void vse8(vr_t vs3, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vse8(vs3, rs1, vm));
    }

    void vse16(vr_t vs3, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vse16(vs3, rs1, vm));
    }

    void vse32(vr_t vs3, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vse32(vs3, rs1, vm));
    }

    void vse64(vr_t vs3, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vse64(vs3, rs1, vm));
    }

    void vse128(vr_t vs3, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vse128(vs3, rs1, vm));
    }

    void vse256(vr_t vs3, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vse256(vs3, rs1, vm));
    }

    void vse512(vr_t vs3, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vse512(vs3, rs1, vm));
    }

    void vse1024(vr_t vs3, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vse1024(vs3, rs1, vm));
    }

    // Vector Load (constant stride)
    template<typename T>
    void vls(vr_t vd, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vls(vd, rs1, rs2, vm, sizeof(T)));
    }

    void vls(vr_t vd, gpr_t rs1, gpr_t rs2, unsigned int sew_bytes,
                                            vmask_t vm = vmask::unmasked) {
        push(rvj_vls(vd, rs1, rs2, vm, sew_bytes));
    }

    void vlse8(vr_t vd, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vlse8(vd, rs1, rs2, vm));
    }

    void vlse16(vr_t vd, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vlse16(vd, rs1, rs2, vm));
    }

    void vlse32(vr_t vd, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vlse32(vd, rs1, rs2, vm));
    }

    void vlse64(vr_t vd, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vlse64(vd, rs1, rs2, vm));
    }

    void vlse128(vr_t vd, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vlse128(vd, rs1, rs2, vm));
    }

    void vlse256(vr_t vd, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vlse256(vd, rs1, rs2, vm));
    }

    void vlse512(vr_t vd, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vlse512(vd, rs1, rs2, vm));
    }

    void vlse1024(vr_t vd, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vlse1024(vd, rs1, rs2, vm));
    }

    // Vector Store (constant stride)
    template<typename T>
    void vss(vr_t vs3, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vss(vs3, rs1, rs2, vm, sizeof(T)));
    }

    void vss(vr_t vs3, gpr_t rs1, gpr_t rs2, unsigned int sew_bytes,
                                            vmask_t vm = vmask::unmasked) {
        push(rvj_vss(vs3, rs1, rs2, vm, sew_bytes));
    }

    void vsse8(vr_t vs3, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsse8(vs3, rs1, rs2, vm));
    }

    void vsse16(vr_t vs3, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsse16(vs3, rs1, rs2, vm));
    }

    void vsse32(vr_t vs3, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsse32(vs3, rs1, rs2, vm));
    }

    void vsse64(vr_t vs3, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsse64(vs3, rs1, rs2, vm));
    }

    void vsse128(vr_t vs3, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsse128(vs3, rs1, rs2, vm));
    }

    void vsse256(vr_t vs3, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsse256(vs3, rs1, rs2, vm));
    }

    void vsse512(vr_t vs3, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsse512(vs3, rs1, rs2, vm));
    }

    void vsse1024(vr_t vs3, gpr_t rs1, gpr_t rs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsse1024(vs3, rs1, rs2, vm));
    }

    // Vector Load (unordered indexed)
    template<typename T>
    void vlux(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vlux(vd, rs1, vs2, vm, sizeof(T)));
    }

    void vlux(vr_t vd, gpr_t rs1, vr_t vs2, unsigned int sew_bytes,
                                            vmask_t vm = vmask::unmasked) {
        push(rvj_vlux(vd, rs1, vs2, vm, sew_bytes));
    }

    void vluxei8(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vluxei8(vd, rs1, vs2, vm));
    }

    void vluxei16(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vluxei16(vd, rs1, vs2, vm));
    }

    void vluxei32(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vluxei32(vd, rs1, vs2, vm));
    }

    void vluxei64(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vluxei64(vd, rs1, vs2, vm));
    }

    void vluxei128(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vluxei128(vd, rs1, vs2, vm));
    }

    void vluxei256(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vluxei256(vd, rs1, vs2, vm));
    }

    void vluxei512(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vluxei512(vd, rs1, vs2, vm));
    }

    void vluxei1024(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vluxei1024(vd, rs1, vs2, vm));
    }

    // Vector Store (unordered indexed)
    template<typename T>
    void vsux(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsux(vs3, rs1, vs2, vm, sizeof(T)));
    }

    void vsux(vr_t vs3, gpr_t rs1, vr_t vs2, unsigned int sew_bytes,
                                            vmask_t vm = vmask::unmasked) {
        push(rvj_vsux(vs3, rs1, vs2, vm, sew_bytes));
    }

    void vsuxei8(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsuxei8(vs3, rs1, vs2, vm));
    }

    void vsuxei16(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsuxei16(vs3, rs1, vs2, vm));
    }

    void vsuxei32(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsuxei32(vs3, rs1, vs2, vm));
    }

    void vsuxei64(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsuxei64(vs3, rs1, vs2, vm));
    }

    void vsuxei128(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsuxei128(vs3, rs1, vs2, vm));
    }

    void vsuxei256(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsuxei256(vs3, rs1, vs2, vm));
    }

    void vsuxei512(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsuxei512(vs3, rs1, vs2, vm));
    }

    void vsuxei1024(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsuxei1024(vs3, rs1, vs2, vm));
    }

    // Vector Load (ordered indexed)
    template<typename T>
    void vlox(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vlox(vd, rs1, vs2, vm, sizeof(T)));
    }

    void vlox(vr_t vd, gpr_t rs1, vr_t vs2, unsigned int sew_bytes,
                                            vmask_t vm = vmask::unmasked) {
        push(rvj_vlox(vd, rs1, vs2, vm, sew_bytes));
    }

    void vloxei8(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vloxei8(vd, rs1, vs2, vm));
    }

    void vloxei16(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vloxei16(vd, rs1, vs2, vm));
    }

    void vloxei32(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vloxei32(vd, rs1, vs2, vm));
    }

    void vloxei64(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vloxei64(vd, rs1, vs2, vm));
    }

    void vloxei128(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vloxei128(vd, rs1, vs2, vm));
    }

    void vloxei256(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vloxei256(vd, rs1, vs2, vm));
    }

    void vloxei512(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vloxei512(vd, rs1, vs2, vm));
    }

    void vloxei1024(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vloxei1024(vd, rs1, vs2, vm));
    }

    // Vector Store (ordered indexed)
    template<typename T>
    void vsox(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsox(vs3, rs1, vs2, vm, sizeof(T)));
    }

    void vsox(vr_t vs3, gpr_t rs1, vr_t vs2, unsigned int sew_bytes,
                                            vmask_t vm = vmask::unmasked) {
        push(rvj_vsox(vs3, rs1, vs2, vm, sew_bytes));
    }

    void vsoxei8(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsoxei8(vs3, rs1, vs2, vm));
    }

    void vsoxei16(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsoxei16(vs3, rs1, vs2, vm));
    }

    void vsoxei32(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsoxei32(vs3, rs1, vs2, vm));
    }

    void vsoxei64(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsoxei64(vs3, rs1, vs2, vm));
    }

    void vsoxei128(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsoxei128(vs3, rs1, vs2, vm));
    }

    void vsoxei256(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsoxei256(vs3, rs1, vs2, vm));
    }

    void vsoxei512(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsoxei512(vs3, rs1, vs2, vm));
    }

    void vsoxei1024(vr_t vs3, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsoxei1024(vs3, rs1, vs2, vm));
    }

    // Vector Unit-stride Fault-Only-First Loads
    template<typename T>
    void vlff(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vlff(vd, rs1, vm, sizeof(T)));
    }

    void vlff(vr_t vd, gpr_t rs1, unsigned int sew_bytes,
                                            vmask_t vm = vmask::unmasked) {
        push(rvj_vlff(vd, rs1, vm, sew_bytes));
    }

    void vle8ff(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle8ff(vd, rs1, vm));
    }

    void vle16ff(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle16ff(vd, rs1, vm));
    }

    void vle32ff(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle32ff(vd, rs1, vm));
    }

    void vle64ff(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle64ff(vd, rs1, vm));
    }

    void vle128ff(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle128ff(vd, rs1, vm));
    }

    void vle256ff(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle256ff(vd, rs1, vm));
    }

    void vle512ff(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle512ff(vd, rs1, vm));
    }

    void vle1024ff(vr_t vd, gpr_t rs1, vmask_t vm = vmask::unmasked) {
        push(rvj_vle1024ff(vd, rs1, vm));
    }

    // Vector Load/Store Segment Instructions (TODO)
    // Vector Load/Store Whole Register Instructions (TODO)
    // Vector AMO Instructions (TODO)
    // Vector Integer Arithmetic Instructions (TODO)

    // Vector Integer Add
    void vadd_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vadd_vv(vd, vs1, vs2, vm));
    }

    void vadd_vx(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vadd_vx(vd, rs1, vs2, vm));
    }

    void vadd_vi(vr_t vd, vr_t vs2, int simm5, vmask_t vm = vmask::unmasked) {
        push(rvj_vadd_vi(vd, vs2, simm5, vm));
    }

    // Vector Integer Subtraction
    void vsub_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsub_vv(vd, vs1, vs2, vm));
    }

    void vsub_vx(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vsub_vx(vd, rs1, vs2, vm));
    }

    // Vector Integer Reverse Subtraction
    void vrsub_vx(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vrsub_vx(vd, rs1, vs2, vm));
    }

    void vrsub_vi(vr_t vd, vr_t vs2, int simm5, vmask_t vm = vmask::unmasked) {
        push(rvj_vrsub_vi(vd, vs2, simm5, vm));
    }

    // Vector Bitwise AND
    void vand_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vand_vv(vd, vs1, vs2, vm));
    }

    void vand_vx(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vand_vx(vd, rs1, vs2, vm));
    }

    void vand_vi(vr_t vd, vr_t vs2, int simm5, vmask_t vm = vmask::unmasked) {
        push(rvj_vand_vi(vd, vs2, simm5, vm));
    }

    // Vector Bitwise OR
    void vor_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vor_vv(vd, vs1, vs2, vm));
    }

    void vor_vx(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vor_vx(vd, rs1, vs2, vm));
    }

    void vor_vi(vr_t vd, vr_t vs2, int simm5, vmask_t vm = vmask::unmasked) {
        push(rvj_vor_vi(vd, vs2, simm5, vm));
    }

    // Vector Bitwise XOR
    void vxor_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vxor_vv(vd, vs1, vs2, vm));
    }

    void vxor_vx(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vxor_vx(vd, rs1, vs2, vm));
    }

    void vxor_vi(vr_t vd, vr_t vs2, int simm5, vmask_t vm = vmask::unmasked) {
        push(rvj_vxor_vi(vd, vs2, simm5, vm));
    }

    // Vector Integer Divide
    void vdiv_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vdiv_vv(vd, vs1, vs2, vm));
    }

    void vdiv_vx(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vdiv_vx(vd, rs1, vs2, vm));
    }

    // Vector Integer Multiply
    void vmul_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vmul_vv(vd, vs1, vs2, vm));
    }

    void vmul_vx(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vmul_vx(vd, rs1, vs2, vm));
    }

    // Vector Integer Fused Multiply and Add (vd is first multiplicand)
    void vmadd_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vmadd_vv(vd, vs1, vs2, vm));
    }

    void vmadd_vx(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vmadd_vx(vd, rs1, vs2, vm));
    }

    // Vector Integer Fused Multiply and Add (vd is addend)
    void vmacc_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vmacc_vv(vd, vs1, vs2, vm));
    }

    void vmacc_vx(vr_t vd, gpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vmacc_vx(vd, rs1, vs2, vm));
    }

    // Vector Fixed-Point Arithmetic Instructions (TODO)
    // Vector Floating-Point Arithmetic Operations (TODO)

    // Vector Floating-point Add
    void vfadd_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfadd_vv(vd, vs1, vs2, vm));
    }

    void vfadd_vf(vr_t vd, fpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfadd_vf(vd, rs1, vs2, vm));
    }

    // Vector Floating-point Subtraction
    void vfsub_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfsub_vv(vd, vs1, vs2, vm));
    }

    void vfsub_vf(vr_t vd, fpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfsub_vf(vd, rs1, vs2, vm));
    }

    // Vector Floating-point Reverse Subtraction
    void vfrsub_vf(vr_t vd, fpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfrsub_vf(vd, rs1, vs2, vm));
    }

    // Vector Floating-point Divide
    void vfdiv_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfdiv_vv(vd, vs1, vs2, vm));
    }

    void vfdiv_vf(vr_t vd, fpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfdiv_vf(vd, rs1, vs2, vm));
    }

    // Vector Floating-point Reverse Divide
    void vfrdiv_vf(vr_t vd, fpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfrdiv_vf(vd, rs1, vs2, vm));
    }

    // Vector Floating-point Multiply
    void vfmul_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfmul_vv(vd, vs1, vs2, vm));
    }

    void vfmul_vf(vr_t vd, fpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfmul_vf(vd, rs1, vs2, vm));
    }

    // Vector Floating-point Fused Multiply and Add (vd is first multiplicand)
    void vfmadd_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfmadd_vv(vd, vs1, vs2, vm));
    }

    void vfmadd_vf(vr_t vd, fpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfmadd_vf(vd, rs1, vs2, vm));
    }

    // Vector Floating-point Fused Multiply and Add (vd is addend)
    void vfmacc_vv(vr_t vd, vr_t vs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfmacc_vv(vd, vs1, vs2, vm));
    }

    void vfmacc_vf(vr_t vd, fpr_t rs1, vr_t vs2, vmask_t vm = vmask::unmasked) {
        push(rvj_vfmacc_vf(vd, rs1, vs2, vm));
    }

    // Vector Reduction Operations (TODO)
    // Vector Mask Instructions (TODO)
    void vmand_vv(rvj_vmask vd, vr_t vs1, vr_t vs2) {
        push(rvj_vmand_vv(vd, vs1, vs2));
    }

    void vmslt_vx(rvj_vmask vd, vr_t vs1, vr_t vs2) {
        push(rvj_vmslt_vx(vd, vs1, vs2));
    }

    void vmseq_vv(rvj_vmask vd, vr_t vs1, vr_t vs2) {
        push(rvj_vmseq_vv(vd, vs1, vs2));
    }
    
    void vmsgt_vx(rvj_vmask vd, vr_t vs1, vr_t vs2) {
        push(rvj_vmsgt_vx(vd, vs1, vs2));
    }

    // Vector Permutation Instructions (TODO)
    // Vector Integer Scalar Move Instructions
    void vmv_xs(gpr_t vd, vr_t vs2) {
        push(rvj_vmv_xs(vd, vs2));
    }

    void vmv_sx(vr_t vd, gpr_t rs1) {
        push(rvj_vmv_sx(vd, rs1));
    }

    // Vector Floating-Point Scalar Move Instructions
    void vfmv_fs(fpr_t vd, vr_t vs2) {
        push(rvj_vfmv_fs(vd, vs2));
    }

    void vfmv_sf(vr_t vd, fpr_t rs1) {
        push(rvj_vfmv_sf(vd, rs1));
    }

    // ... (TODO) ...
    // Whole Vector Register Move
    void vmvr_v(vr_t vd, vr_t vs2, unsigned int nr) {
        push(rvj_vmvr_v(vd, vs2, nr));
    }

    void vmv1r_v(vr_t vd, vr_t vs2) {
        push(rvj_vmv1r_v(vd, vs2));
    }

    void vmv2r_v(vr_t vd, vr_t vs2) {
        push(rvj_vmv2r_v(vd, vs2));
    }

    void vmv4r_v(vr_t vd, vr_t vs2) {
        push(rvj_vmv4r_v(vd, vs2));
    }

    void vmv8r_v(vr_t vd, vr_t vs2) {
        push(rvj_vmv8r_v(vd, vs2));
    }


    /**************************************************************************
    ** RV64-V Standard Extension Pseudo Instructions
    **************************************************************************/
    void vsetvlmax(gpr_t rd, gpr_t rs2) {
        push(rvj_vsetvlmax(rd, rs2));
    }

    void vsetvlmaxi(gpr_t rd, unsigned int vtypei) {
        push(rvj_vsetvlmaxi(rd, vtypei));
    }
};

}