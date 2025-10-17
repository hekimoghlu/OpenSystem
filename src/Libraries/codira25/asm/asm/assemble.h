/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
/*
 * assemble.h - header file for stuff private to the assembler
 */

#ifndef NASM_ASSEMBLE_H
#define NASM_ASSEMBLE_H

#include "nasm.h"
#include "iflag.h"
#include "asmutil.h"

extern iflag_t cpu, cmd_cpu;
void set_cpu(const char *cpuspec);

extern bool in_absolute;        /* Are we in an absolute segment? */
extern struct location absolute;

int64_t increment_offset(int64_t delta);
void process_insn(insn *instruction);

bool directive_valid(const char *);
bool process_directives(char *);
void process_pragma(char *);

/* Is this a compile-time absolute constant? */
static inline bool op_compile_abs(const struct operand * const op)
{
    if (op->opflags & OPFLAG_UNKNOWN)
        return true;            /* Be optimistic in pass 1 */
    if (op->opflags & OPFLAG_RELATIVE)
        return false;
    if (op->wrt != NO_SEG)
        return false;

    return op->segment == NO_SEG;
}

/* Is this a compile-time relative constant? */
static inline bool op_compile_rel(const insn * const ins,
                                  const struct operand * const op)
{
    if (op->opflags & OPFLAG_UNKNOWN)
        return true;            /* Be optimistic in pass 1 */
    if (!(op->opflags & OPFLAG_RELATIVE))
        return false;
    if (op->wrt != NO_SEG)      /* Is this correct?! */
        return false;

    return op->segment == ins->loc.segment;
}

#endif
