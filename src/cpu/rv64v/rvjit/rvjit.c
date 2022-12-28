#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "instruction_types.h"

#define INSTR_ALLOC 128
#define LABEL_ALLOC 8
#define LABELREF_ALLOC 16
#define LABEL_NAME_SIZE 28

# ifdef __MAP_ANONYMOUS
#  define MAP_ANONYMOUS	__MAP_ANONYMOUS	/* Don't use a file.  */
# else
#  define MAP_ANONYMOUS	0x20		/* Don't use a file.  */
# endif

/// @brief Check if EXPR results in rvj_success, otherwise forward the error.
/// @param EXPR A expression returning a rvj_status_t
#define CHECK(EXPR)\
do { rvj_status_t ret = EXPR; if (ret != rvj_success) return ret; } while (0);

/// @brief Extend the capacity of an array by AEXT if the array is full.
/// @param A The array variable
/// @param DTYPE Array data type
/// @param ASIZE Array occupancy
/// @param ACAP Array capacity
/// @param AEXT A number for how much to extend the capacity
#define EXTEND_ARRAY(A, DTYPE, ASIZE, ACAP, AEXT)\
if ((ASIZE+1) > ACAP) {\
    void *new_buffer;\
    ACAP += AEXT;\
    new_buffer = malloc(sizeof(DTYPE) * ACAP);\
    if (!new_buffer)\
        return rvj_out_of_memory;\
    memcpy(new_buffer, A, sizeof(DTYPE) * ASIZE);\
    free(A);\
    A = new_buffer;\
}

/// @brief Allocate memory on @ref v to store @ref n elements of type @ref t.
/// @param v The target variable
/// @param t The datatype
/// @param n The number of elements to allocate space for
#define TALLOC(v, t, n)\
v = malloc(sizeof(t) * n); if (!v) return rvj_out_of_memory;

typedef struct rvj_label {
    char name[LABEL_NAME_SIZE];
    int instr_id;
} rvj_label;

typedef struct rvj_labelref {
    int label_id;
    int instr_id;
} rvj_labelref;

typedef struct rvj_function {
    void *ptr;
    size_t size;
} rvj_function;

/// @struct rvj_asm
/// @brief Structure to manage the list of instructions, labels, references
/// and overall state during code generation.
/// @details This structure allocates the function handle but is not its owner.
/// The JIT must only free the resources of a function handle if this handle
/// was never requested by any user at the time of the JIT handle deletion.
typedef struct rvj_asm {
    int complete;            /// Code listing status
    rvj_function *fhandle;   /// Pointer to the function handle
    rvj_label *labels;       /// List of labels
    int n_labels;            /// Amount of listed labels
    int cap_label;           /// Capacity for the list of labels
    rvj_instr *instrs;       /// List of instructions
    int n_instr;             /// Amount of listed instructions
    int cap_instr;           /// Capacity for the list of instructions
    rvj_labelref *labelrefs; /// List of label references
    int n_labelrefs;         /// Amount of listed label references
    int cap_labelref;        /// Capacity for the list of label references
    int fhandle_accessed;    /// Toggle to delete function if unused
} rvj_asm;

rvj_status_t rvj_asm_init(rvj_asm **h) {
    rvj_asm *out = malloc(sizeof(rvj_asm));
    out->fhandle = NULL;
    out->complete = 0;
    out->n_instr = 0;
    out->n_labels = 0;
    out->n_labelrefs = 0;
    out->fhandle_accessed = 0;
    out->cap_instr = INSTR_ALLOC;
    out->cap_label = LABEL_ALLOC;
    out->cap_labelref = LABELREF_ALLOC;
    TALLOC(out->instrs, rvj_instr, out->cap_instr);
    TALLOC(out->labels, rvj_label, out->cap_label);
    TALLOC(out->labelrefs, rvj_labelref, out->cap_labelref);
    *h = out;
    return rvj_success;
}

void rvj_asm_free(rvj_asm **h) {
    if (!h || !*h)
        return;
    if ((*h)->labels)
        free((*h)->labels);
    if ((*h)->instrs)
        free((*h)->instrs);
    if ((*h)->labelrefs)
        free((*h)->labelrefs);
    if (!(*h)->fhandle_accessed && (*h)->fhandle) {
        free((*h)->fhandle);
        (*h)->fhandle = NULL;
    }
    free(*h);
    *h = NULL;
}

rvj_status_t rvj_asm_label(rvj_asm *h, const char* name) {
    int label_id = -1;
    rvj_label *label;
    if (h->complete)
        return rvj_invalid_arguments;
    // check array capacity and extend if necessary
    EXTEND_ARRAY(h->labels, rvj_label, h->n_labels, h->cap_label, LABEL_ALLOC);
    // check if label was already defined (forward reference)
    for (int i = 0; i < h->n_labels; ++i) {
        if (strncmp(h->labels[i].name, name, LABEL_NAME_SIZE) == 0) {
            label_id = i;
            break;
        }
    }
    if (label_id < 0) {
        // add the new entry to the array
        label = &h->labels[h->n_labels++];
        strncpy(label->name, name, LABEL_NAME_SIZE-1);
    } else {
        // reference existing label
        label = &h->labels[label_id];
    }
    label->instr_id = h->n_instr;
    return rvj_success;
}

rvj_status_t rvj_asm_push(rvj_asm *h, rvj_instr w) {
    if (h->complete)
        return rvj_invalid_arguments;
    // check instruction array capacity and extend if necessary
    EXTEND_ARRAY(h->instrs, rvj_instr, h->n_instr, h->cap_instr, INSTR_ALLOC);
    // add the new instruction to the array
    h->instrs[h->n_instr++] = w;
    return rvj_success;
}

rvj_status_t rvj_asm_push_lref(rvj_asm *h, rvj_instr w, const char * name) {
    if (h->complete)
        return rvj_invalid_arguments;
    // add the instruction and save its index for later processing
    CHECK(rvj_asm_push(h, w));
    int const instruction_index = h->n_instr-1;
    // Find the label or create one if it did not exist (forward declaration)
    int label_id = -1;
    for (int i = 0; i < h->n_labels; ++i) {
        if (strncmp(name, h->labels[i].name, LABEL_NAME_SIZE) == 0) {
            label_id = i;
            break;
        }
    }
    if (label_id < 0) {
        CHECK(rvj_asm_label(h, name));
        label_id = h->n_labels-1;
        h->labels[label_id].instr_id = -1; // Mark new label position as undef
    }
    // check label references array capacity and extend if necessary
    EXTEND_ARRAY(h->labelrefs, rvj_labelref, h->n_labelrefs,
                 h->cap_labelref, LABELREF_ALLOC);
    rvj_labelref *labelref = &h->labelrefs[h->n_labelrefs++];
    labelref->label_id = label_id;
    labelref->instr_id = instruction_index;
    return rvj_success;
}

rvj_status_t rvj_asm_done(rvj_asm *h) {
    if (h->complete)
        return rvj_invalid_arguments;
    if (h->n_instr == 0)
        return rvj_empty;
    h->complete = 1;
    for (int i = 0; i < h->n_labelrefs; ++i) {
        rvj_labelref *labelref = &h->labelrefs[i];
        int const iid = labelref->instr_id;
        int const lid = labelref->label_id;
        if (iid < 0 || iid >= h->n_instr)   // Instruction index corrupted
            return rvj_error;
        if (lid < 0 || lid >= h->n_labels)  // Label name undefined
            return rvj_undefined_label;
        int const lpos = h->labels[lid].instr_id;
        if (lpos < 0 || lpos >= h->n_instr) // Label position undefined
            return rvj_undefined_label;
        rvj_instr *instr = &h->instrs[iid];
        // Calculate the offset in bytes
        int const offset = lpos - iid;
        // All branch instructions use the B-type instruction format;
        // The 12-bit immediate encodes signed offsets in multiples of 2 bytes;
        rvj_reg imm = rvj_is_type_b(*instr) ? imm_type_b(offset << 1)
        // The jump and link (JAL) instruction uses the J-type format, where
        // the J-immediate encodes a signed offset in multiples of 2 bytes.
            : rvj_is_jal(*instr) ? imm_type_j(offset << 1)
        // The indirect jump instruction JALR (jump and link register) uses
        // the I-type encoding (the offset is in bytes).
            : imm_type_i(offset << 2);
        *instr |= imm;
    }
    TALLOC(h->fhandle, rvj_function, 1);
    h->fhandle->size = sizeof(rvj_instr) * h->n_instr;
    h->fhandle->ptr = mmap(NULL, h->fhandle->size,
            PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    memcpy(h->fhandle->ptr, h->instrs, h->fhandle->size);
    mprotect(h->fhandle->ptr, h->fhandle->size, PROT_READ | PROT_EXEC);
    return rvj_success;
}

rvj_status_t rvj_asm_get_function_handle(rvj_asm_t h, rvj_function_t *out) {
    *out = NULL;
    if (!(h->complete) || !(h->fhandle))
        return rvj_invalid_arguments;
    *out = h->fhandle;
    h->fhandle_accessed = 1; // Pass the ownership to whatever entity called
    return rvj_success;
}

size_t rvj_function_get_size(const rvj_function_t h) {
    return h->size;
}

size_t rvj_function_dump(const rvj_function_t h, char * const out) {
    char *src = (char*) h->ptr;
    for (int i = 0; i < h->size; ++i)
        out[i] = src[i];
    return rvj_function_get_size(h);
}

void rvj_function_get_pointer(const rvj_function_t h, void **out) {
    *out = h->ptr;
}

void rvj_function_free(rvj_function **h) {
    if (!h || !(*h))
        return;
    if ((*h)->ptr)
        munmap((*h)->ptr, (*h)->size);
    (*h)->ptr = NULL;
    (*h)->size = 0;
    free(*h);
    *h = NULL;
}

#undef CHECK
#undef EXTEND_ARRAY
#undef TALLOC
