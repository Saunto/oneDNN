#ifndef CPU_RV64V_JIT_JIT_ASSEMBLER
#define CPU_RV64V_JIT_JIT_ASSEMBLER

#include "cpu/rv64v/rvjit/rvjit.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

class jit_assembler : public rvjit::assembler {
public:
    jit_assembler() : rvjit::assembler() {}

    /// The minimum integer represented as a 12-bit immediate (inclusive)
    int imm_min() const { return -0x800; }

    /// The maximum integer represented as a 12-bit immediate (non-inclusive)
    int imm12_max() const { return 0x800; }

    /// Check if imm is representable as a 12-bit immediate
    bool can_be_imm12(const long int &imm) const {
        return imm >= imm_min() && imm < imm12_max();
    }

    /// Get the 20 most significat bits of a 32-bit integer
    int upper20(const int &imm) const {
        return (imm >> 12) & 0xFFFFF;
    }

    /// Get the 12 least significant bits of a 32-bit integer
    int lower12(const int &imm) const {
        return imm & 0xFFF;
    }

    /// Loads a 32-bit constant to the x register
    void load_constant(const rvjit::gpr_t x, int c) {
        if (can_be_imm12(c)) {
            li(x, c);
            return;
        }
        int lower = lower12(c);
        if (lower >= imm12_max()) {
            // Most significant bit is 1, and the immediate would be
            // interpreted as negative!
            lui(x, upper20(c+imm12_max()));
            addi(x, x, lower);
        } else {
            lui(x, upper20(c));
            if (lower)
                ori(x, x, lower);
        }
    }

    void add_constant(const rvjit::gpr_t rd, const rvjit::gpr_t rs1,
                      const rvjit::gpr_t tmp, int c) {
        if (rd == rs1 && c == 0)
            return;
        if (can_be_imm12(c)) {
            addi(rd, rs1, c);
        } else {
            load_constant(tmp, c);
            add(rd, rs1, tmp);
        }
    }

    struct register_pool_t {
    private:
        rvjit::gpr_t _pool[32];
        int _size;
        int _position;
    
    public:
        register_pool_t() : _size(0), _position(0) {}

        register_pool_t(std::initializer_list<rvjit::gpr_t> regs) {
            _size = regs.size();
            _position = 0;
            int i = 0;
            for (auto r : regs)
                _pool[i++] = r;
        }

        int position() const { return _position; }
        int size() const { return _size; }
        rvjit::gpr_t head() const { return _pool[_position]; }

        rvjit::gpr_t pick() {
            assert(_position+1 < _size);
            return _pool[_position++];
        }

        void reset() { _position = 0; }
    };

    struct assembly_constant_t {
        rvjit::gpr_t rd;    // Register used if value is not immediate
        int value;          // The constant numerical value
        bool is_imm;        // If the value is an immediate
        bool is_in_reg;     // If the value is in the register

        assembly_constant_t() : rd(rvj_x0), value(0), is_imm(false) {}

        assembly_constant_t(rvjit::gpr_t r, int v, bool i)
            : rd(r), value(v), is_imm(i) {}


        /// Checs if the constant is ready for use in the assembler
        /// @details immediates are always ready 
        bool is_ready() const { return is_imm || is_in_reg; }
    };

    assembly_constant_t asm_const(rvjit::gpr_t reg, int value,
                                        bool force_register = false) const {
        auto const is_imm = can_be_imm12(value) && !force_register;
        return assembly_constant_t(reg, value, is_imm);
    }

    assembly_constant_t asm_const(register_pool_t &rpool, int value,
                                        bool force_register = false) const {
        auto const is_imm = can_be_imm12(value) && !force_register;
        if (is_imm)
            return assembly_constant_t(rpool.head(), value, is_imm);
        else
            return assembly_constant_t(rpool.pick(), value, is_imm);
    }

    void prepare_constant(assembly_constant_t &c) {
        if (!c.is_ready()) {
            load_constant(c.rd, c.value);
            c.is_in_reg = true;
        }
    }

    void add_constant(const rvjit::gpr_t rd, const rvjit::gpr_t rs1,
                        assembly_constant_t &c) {
        if (rd == rs1 && c.value == 0)
            return;
        if (c.is_ready() && !c.is_imm) {
            add(rd, rs1, c.rd);
        } else {
            prepare_constant(c);
            add_constant(rd, rs1, c.rd, c.value);
        }
    }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64V_JIT_JIT_ASSEMBLER