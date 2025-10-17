/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#ifndef	_ARCH_ARM_REG_HELP_H_
#define	_ARCH_ARM_REG_HELP_H_

/* Bitfield definition aid */
#define	BITS_WIDTH(msb, lsb)	((msb)-(lsb)+1)
#define	BIT_WIDTH(pos)		(1)	/* mostly to record the position */

/* Mask creation */
#define	MKMASK(width, offset)	(((unsigned)-1)>>(32-(width))<<(offset))
#define	BITSMASK(msb, lsb)	MKMASK(BITS_WIDTH(msb, lsb), lsb & 0x1f)
#define	BITMASK(pos)		MKMASK(BIT_WIDTH(pos), pos & 0x1f)

/* Register addresses */
#if	__ASSEMBLER__
# define	REG_ADDR(type, addr)	(addr)
#else	/* __ASSEMBLER__ */
# define	REG_ADDR(type, addr)	(*(volatile type *)(addr))
#endif	/* __ASSEMBLER__ */

/* Cast a register to be an unsigned */
#if defined(__arm__)
#define	CONTENTS(foo)	(*(unsigned*) &(foo))
/* Stack pointer must always be a multiple of 4 */
#define	STACK_INCR	4
#elif defined(__arm64__)
#define	CONTENTS(foo)	(*(unsigned long long*) &(foo))
/* Stack pointer must always be a multiple of 16 */
#define	STACK_INCR	16
#else /* !defined(__arm__) && !defined(__arm64__) */
#error Unknown architecture.
#endif /* !defined(__arm__) && !defined(__arm64__) */

#define	ROUND_FRAME(x)	((((unsigned)(x)) + STACK_INCR - 1) & ~(STACK_INCR-1))

/* STRINGIFY -- perform all possible substitutions, then stringify */
#define	__STR(x)	#x		/* just a helper macro */
#define	STRINGIFY(x)	__STR(x)

#endif	/* _ARCH_ARM_REG_HELP_H_ */
