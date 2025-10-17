/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#ifndef YASM_LC3BARCH_H
#define YASM_LC3BARCH_H

/* Types of immediate.  All immediates are stored in the LSBs of the insn. */
typedef enum lc3b_imm_type {
    LC3B_IMM_NONE = 0,  /* no immediate */
    LC3B_IMM_4,         /* 4-bit */
    LC3B_IMM_5,         /* 5-bit */
    LC3B_IMM_6_WORD,    /* 6-bit, word-multiple (byte>>1) */
    LC3B_IMM_6_BYTE,    /* 6-bit, byte-multiple */
    LC3B_IMM_8,         /* 8-bit, word-multiple (byte>>1) */
    LC3B_IMM_9,         /* 9-bit, signed, word-multiple (byte>>1) */
    LC3B_IMM_9_PC       /* 9-bit, signed, word-multiple, PC relative */
} lc3b_imm_type;

/* Bytecode types */

typedef struct lc3b_insn {
    yasm_value imm;             /* immediate or relative value */
    lc3b_imm_type imm_type;     /* size of the immediate */

    unsigned int opcode;        /* opcode */
} lc3b_insn;

void yasm_lc3b__bc_transform_insn(yasm_bytecode *bc, lc3b_insn *insn);

yasm_arch_insnprefix yasm_lc3b__parse_check_insnprefix
    (yasm_arch *arch, const char *id, size_t id_len, unsigned long line,
     /*@out@*/ /*@only@*/ yasm_bytecode **bc, /*@out@*/ uintptr_t *prefix);
yasm_arch_regtmod yasm_lc3b__parse_check_regtmod
    (yasm_arch *arch, const char *id, size_t id_len,
     /*@out@*/ uintptr_t *data);

int yasm_lc3b__intnum_tobytes
    (yasm_arch *arch, const yasm_intnum *intn, unsigned char *buf,
     size_t destsize, size_t valsize, int shift, const yasm_bytecode *bc,
     int warn);

/*@only@*/ yasm_bytecode *yasm_lc3b__create_empty_insn(yasm_arch *arch,
                                                       unsigned long line);

void yasm_lc3b__ea_destroy(/*@only@*/ yasm_effaddr *ea);

#endif
