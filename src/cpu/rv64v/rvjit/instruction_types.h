#ifndef INSTRUCTION_TYPES_H
#define INSTRUCTION_TYPES_H

#include "rvjit.h"

/// @brief A mask embedded with an immediate value for I-type instructions
rvj_instr imm_type_i(int imm);

/// @brief A mask embedded with an immediate value for S-type instructions
rvj_instr imm_type_s(int imm);

/// @brief A mask embedded with an immediate value for B-type instructions
rvj_instr imm_type_b(int imm);

/// @brief A mask embedded with an immediate value for U-type instructions
rvj_instr imm_type_u(int imm);

/// @brief A mask embedded with an immediate value for J-type instructions
rvj_instr imm_type_j(int imm);

/// @brief Checks if the instruction conforms to the B-type instruction format
/// @return 1 if it conforms, 0 otherwise
/// @details Checks if the opcode matches the BRANCH opcode, shared across all
/// B-type instructions
int rvj_is_type_b(rvj_instr i);

/// @brief Checks if the instruction is a jump and link (JAL)
/// @return 1 if it conforms, 0 otherwise
/// @details Checks if the opcode matches the JAL opcode
int rvj_is_jal(rvj_instr i);

/// @brief Creates an R-type instruction from its bitfields
rvj_instr opR(int rd, int rs1, int rs2, int op, int f3, int f7);

/// @brief Creates an R4-type instruction from its bitfields
rvj_instr opR4(int rd, int rs1, int rs2, int rs3, int op, int f3, int f2);

/// @brief Creates an I-type instruction from its bitfields
rvj_instr opI(int rd, int rs1, int imm, int op, int f3);

/// @brief Creates a S-type instruction from its bitfields
rvj_instr opS(int rs1, int rs2, int imm, int op, int f3);

/// @brief Creates a B-type instruction from its bitfields
rvj_instr opB(int rs1, int rs2, int imm, int f3);

/// @brief Creates an U-type instruction from its bitfields
rvj_instr opU(int rd, int imm, int op);

/// @brief Creates a J-type instruction from its bitfields
rvj_instr opJ(int rd, int imm, int op);

/// @brief Creates an FP-type instruction from its bitfields
rvj_instr opFP(int rd, int rs1, int rs2, int f5, int fmt, rvj_rm rm);

/// @brief Creates a V-type instruction from its bitfields
rvj_instr opV(int vd, int vs1, int vs2, int f6, int width, int vm);

/// @brief Creates a int vector-vector V-type instruction from its bitfields
rvj_instr opIVV(int vd, int vs1, int vs2, int f6, int vm);

/// @brief Creates a float vector-vector V-type instruction from its bitfields
rvj_instr opFVV(int vd, int vs1, int vs2, int f6, int vm);

/// @brief Creates a M vector-vector V-type instruction from its bitfields
rvj_instr opMVV(int vd, int vs1, int vs2, int f6, int vm);

/// @brief Creates a vector-immediate V-type instruction from its bitfields
rvj_instr opIVI(int vd, int simm5, int vs2, int f6, int vm);

/// @brief Creates a vector-scalar (GPR) V-type instruction from its bitfields
rvj_instr opIVX(int vd, int rs1, int vs2, int f6, int vm);

/// @brief Creates a vector-scalar (FP) V-type instruction from its bitfields
rvj_instr opFVF(int vd, int rs1, int vs2, int f6, int vm);

/// @brief Creates a vector-scalar (GPR) V-type instruction from its bitfields
rvj_instr opMVX(int vd, int rs1, int vs2, int f6, int vm);

#endif