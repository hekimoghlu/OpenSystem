/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
#ifndef _MACHINE_STATIC_IF_H
#error "do not include this file directly, use <machine/static_if.h>"
#else

#define STATIC_IF_RELATIVE      0
#define STATIC_IF_INSN_SIZE     5

typedef long static_if_offset_t;

struct static_if_entry {
	static_if_offset_t      sie_base;
	static_if_offset_t      sie_target;
	unsigned long           sie_link;
};

/* generates a struct static_if_entry */
#define STATIC_IF_ENTRY(n) \
	".pushsection " STATIC_IF_SEGSECT ",regular,live_support"       "\n\t" \
	".align 3"                                                      "\n\t" \
	".quad 1b"                                                      "\n\t" \
	".quad %l1"                                                     "\n\t" \
	".quad _" #n "_jump_key + %c0"                                  "\n\t" \
	".popsection"

/* From "Recommended Multi-Byte Sequence of NOP Instruction" */
#define STATIC_IF_NOP(n, label) \
	asm goto("1: .byte 0x0F,0x1F,0x44,0x00,0x00"                    "\n\t" \
	    STATIC_IF_ENTRY(n) : : "i"(0) : : label)

/* 32-bit jump */
#define STATIC_IF_BRANCH(n, label) \
	asm goto("1: .byte 0xE9; .long %l1 - 2f; 2:"                    "\n\t" \
	    STATIC_IF_ENTRY(n) : : "i"(1) : : label)

#endif /* _MACHINE_STATIC_IF_H */
