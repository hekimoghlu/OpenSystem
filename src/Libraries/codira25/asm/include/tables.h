/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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
 * tables.h
 *
 * Declarations for auto-generated tables
 */

#ifndef NASM_TABLES_H
#define NASM_TABLES_H

#include "compiler.h"
#include "insnsi.h"		/* For enum opcode */

/* --- From insns.dat via insns.pl: --- */

/* insnsn.c */
extern const char * const nasm_insn_names[];

/* --- From regs.dat via regs.pl: --- */

/* regs.c */
extern const char * const nasm_reg_names[];
/* regflags.c */
typedef uint64_t opflags_t;
typedef uint16_t  decoflags_t;
extern const opflags_t nasm_reg_flags[];
/* regvals.c */
extern const int nasm_regvals[];

#endif /* NASM_TABLES_H */
