/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
#ifndef _X86_64_DWARF_UNWIND_H_
#define _X86_64_DWARF_UNWIND_H_

/*
 * This file contains the architecture specific DWARF definitions needed for unwind
 * information added to trap handlers.
 */

/* DWARF Register numbers for x86 */

#define DWARF_RAX 0
#define DWARF_RDX 1
#define DWARF_RCX 2
#define DWARF_RBX 3
#define DWARF_RSI 4
#define DWARF_RDI 5
#define DWARF_RBP 6
#define DWARF_RSP 7
#define DWARF_R8  8
#define DWARF_R9  9
#define DWARF_R10 10
#define DWARF_R11 11
#define DWARF_R12 12
#define DWARF_R13 13
#define DWARF_R14 14
#define DWARF_R15 15
#define DWARF_RIP 16

/* Dwarf opcodes */

#define DW_OP_breg15      0x7f
#define DW_CFA_expression 0x10

/* Convenient DWARF expression macros */

#define DW_FORM_LEN_TWO_BYTE_SLEB 3
#define DW_FORM_LEN_ONE_BYTE_SLEB 2

/* Additional constants for register offsets in the saved state that need to be expressed as SLEB128 */

#define R64_RAX_SLEB128 0x88, 0x01
#define R64_RCX_SLEB128 0x80, 0x01
#define R64_RBX_SLEB128 0xf8, 0x00
#define R64_RBP_SLEB128 0xf0, 0x00
#define R64_RSP_SLEB128 0xd0, 0x01
#define R64_R11_SLEB128 0xe8, 0x00
#define R64_R12_SLEB128 0xe0, 0x00
#define R64_R13_SLEB128 0xd8, 0x00
#define R64_R14_SLEB128 0xd0, 0x00
#define R64_R15_SLEB128 0xc8, 0x00
#define R64_RIP_SLEB128 0xb8, 0x01

/* The actual unwind directives added to trap handlers to let the debugger know where the register state is stored */

/* Unwind Prologue added to each function to indicate the start of the unwind information. */

#define UNWIND_PROLOGUE \
.cfi_sections .eh_frame ;\
.cfi_startproc;        ;\
.cfi_signal_frame       ;\


/* Unwind Epilogue added to each function to indicate the end of the unwind information */

#define UNWIND_EPILOGUE .cfi_endproc


#define UNWIND_DIRECTIVES \
.cfi_escape DW_CFA_expression, DWARF_RAX, DW_FORM_LEN_TWO_BYTE_SLEB, DW_OP_breg15, R64_RAX_SLEB128 ;\
.cfi_escape DW_CFA_expression, DWARF_RDX, DW_FORM_LEN_ONE_BYTE_SLEB, DW_OP_breg15, R64_RDX         ;\
.cfi_escape DW_CFA_expression, DWARF_RCX, DW_FORM_LEN_TWO_BYTE_SLEB, DW_OP_breg15, R64_RCX_SLEB128 ;\
.cfi_escape DW_CFA_expression, DWARF_RBX, DW_FORM_LEN_TWO_BYTE_SLEB, DW_OP_breg15, R64_RBX_SLEB128 ;\
.cfi_escape DW_CFA_expression, DWARF_RSI, DW_FORM_LEN_ONE_BYTE_SLEB, DW_OP_breg15, R64_RSI         ;\
.cfi_escape DW_CFA_expression, DWARF_RDI, DW_FORM_LEN_ONE_BYTE_SLEB, DW_OP_breg15, R64_RDI         ;\
.cfi_escape DW_CFA_expression, DWARF_RBP, DW_FORM_LEN_TWO_BYTE_SLEB, DW_OP_breg15, R64_RBP_SLEB128 ;\
.cfi_escape DW_CFA_expression, DWARF_RSP, DW_FORM_LEN_TWO_BYTE_SLEB, DW_OP_breg15, R64_RSP_SLEB128 ;\
.cfi_escape DW_CFA_expression, DWARF_R8,  DW_FORM_LEN_ONE_BYTE_SLEB, DW_OP_breg15, R64_R8          ;\
.cfi_escape DW_CFA_expression, DWARF_R9,  DW_FORM_LEN_ONE_BYTE_SLEB, DW_OP_breg15, R64_R9          ;\
.cfi_escape DW_CFA_expression, DWARF_R10, DW_FORM_LEN_ONE_BYTE_SLEB, DW_OP_breg15, R64_R10         ;\
.cfi_escape DW_CFA_expression, DWARF_R11, DW_FORM_LEN_TWO_BYTE_SLEB, DW_OP_breg15, R64_R11_SLEB128 ;\
.cfi_escape DW_CFA_expression, DWARF_R12, DW_FORM_LEN_TWO_BYTE_SLEB, DW_OP_breg15, R64_R12_SLEB128 ;\
.cfi_escape DW_CFA_expression, DWARF_R13, DW_FORM_LEN_TWO_BYTE_SLEB, DW_OP_breg15, R64_R13_SLEB128 ;\
.cfi_escape DW_CFA_expression, DWARF_R14, DW_FORM_LEN_TWO_BYTE_SLEB, DW_OP_breg15, R64_R14_SLEB128 ;\
.cfi_escape DW_CFA_expression, DWARF_R15, DW_FORM_LEN_TWO_BYTE_SLEB, DW_OP_breg15, R64_R15_SLEB128 ;\
.cfi_escape DW_CFA_expression, DWARF_RIP, DW_FORM_LEN_TWO_BYTE_SLEB, DW_OP_breg15, R64_RIP_SLEB128 ;\

#endif /*  _X86_64_DWARF_UNWIND_H_ */
