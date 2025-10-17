/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef NASM_INSNS_H
#define NASM_INSNS_H

#include "nasm.h"
#include "tokens.h"
#include "iflag.h"

struct itemplate {
    enum opcode     opcode;             /* the token, passed from "parser.c" */
    int             operands;           /* number of operands */
    opflags_t       opd[MAX_OPERANDS];  /* bit flags for operand types */
    decoflags_t     deco[MAX_OPERANDS]; /* bit flags for operand decorators */
    const uint8_t   *code;              /* the code it assembles to */
    uint32_t        iflag_idx;          /* some flags referenced by index */
};

/* Use this helper to test instruction template flags */
static inline bool itemp_has(const struct itemplate *itemp, unsigned int bit)
{
    return iflag_test(&insns_flags[itemp->iflag_idx], bit);
}

/* Disassembler table structure */

/* Instruction tables for the assembler */
struct itemplate_list {
    unsigned int ntemp;
    const struct itemplate *temp;
};
extern const struct itemplate_list nasm_instructions[];

/* Instruction tables for the disassembler */
extern const struct itemplate * const * const * const ndisasm_itable[];

/* Common table for the byte codes */
extern const uint8_t nasm_bytecodes[];

/*
 * Pseudo-op tests
 */
/* DB-type instruction (DB, DW, ...) */
static inline bool const_func opcode_is_db(enum opcode opcode)
{
    return opcode >= I_DB && opcode < I_RESB;
}

/* RESB-type instruction (RESB, RESW, ...) */
static inline bool const_func opcode_is_resb(enum opcode opcode)
{
    return opcode >= I_RESB && opcode < I_INCBIN;
}

/* Width of Dx and RESx instructions */

/*
 * initialized data bytes length from opcode
 */
static inline int const_func db_bytes(enum opcode opcode)
{
    switch (opcode) {
    case I_DB:
        return 1;
    case I_DW:
        return 2;
    case I_DD:
        return 4;
    case I_DQ:
        return 8;
    case I_DT:
        return 10;
    case I_DO:
        return 16;
    case I_DY:
        return 32;
    case I_DZ:
        return 64;
    case I_none:
        return -1;
    default:
        return 0;
    }
}

/*
 * Uninitialized data bytes length from opcode
 */
static inline int const_func resb_bytes(enum opcode opcode)
{
    switch (opcode) {
    case I_RESB:
        return 1;
    case I_RESW:
        return 2;
    case I_RESD:
        return 4;
    case I_RESQ:
        return 8;
    case I_REST:
        return 10;
    case I_RESO:
        return 16;
    case I_RESY:
        return 32;
    case I_RESZ:
        return 64;
    case I_none:
        return -1;
    default:
        return 0;
    }
}

#endif /* NASM_INSNS_H */
