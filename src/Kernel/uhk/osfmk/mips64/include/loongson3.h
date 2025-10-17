/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
 * Copyright (c) 2016 Visa Hankala
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef _MIPS64_LOONGSON3_H_
#define _MIPS64_LOONGSON3_H_

/*
 * Definitions for Loongson 3A.
 */

#define LS3_CFG_BASE(node)		(0x100000004000ull*(node) + 0x3ff00000)
#define LS3_MEM_BASE(node)		(0x100000000000ull*(node))

#define LS3_IPI_BASE(n, c)		(LS3_CFG_BASE(n) + 0x1000 + 0x100*(c))
#define LS3_IPI_ISR			0x00
#define LS3_IPI_IMR			0x04
#define LS3_IPI_SET			0x08
#define LS3_IPI_CLEAR			0x0c
#define LS3_IPI_MBOX0			0x20
#define LS3_IPI_MBOX1			0x28
#define LS3_IPI_MBOX2			0x30
#define LS3_IPI_MBOX3			0x38

static inline uint32_t
loongson3_get_cpuid(void)
{
	uint32_t tmp;

	asm volatile (
	"	.set push\n"
	"	.set mips64\n"
	"	mfc0 %0, $15, 1\n"	/* EBase */
	"	.set pop\n"
	: "=r" (tmp));

	return tmp & 0xf;
}

#define LS3_COREID(cpuid) ((cpuid) & 3)
#define LS3_NODEID(cpuid) ((cpuid) >> 2)

/*
 * Interrupt router registers
 */

#define LS3_IRT_ENTRY(node, irq)	(LS3_CFG_BASE(node) + 0x1400 + (irq))
#define LS3_IRT_INTISR(node)		(LS3_CFG_BASE(node) + 0x1420)
#define LS3_IRT_INTEN(node)		(LS3_CFG_BASE(node) + 0x1424)
#define LS3_IRT_INTENSET(node)		(LS3_CFG_BASE(node) + 0x1428)
#define LS3_IRT_INTENCLR(node)		(LS3_CFG_BASE(node) + 0x142c)
#define LS3_IRT_INTISR_CORE(node, cpu)	(LS3_CFG_BASE(node) + 0x1440 + (cpu)*8)

/* sys int 0-3 */
#define LS3_IRT_ENTRY_INT(node, x)	LS3_IRT_ENTRY((node), (x))
/* PCI int 0-3 */
#define LS3_IRT_ENTRY_PCI(node, x)	LS3_IRT_ENTRY((node), 0x04+(x))
/* LPC int */
#define LS3_IRT_ENTRY_LPC(node)		LS3_IRT_ENTRY((node), 0x0a)
/* HT0 int 0-7 */
#define LS3_IRT_ENTRY_HT0(node, x)	LS3_IRT_ENTRY((node), 0x10+(x))
/* HT1 int 0-7 */
#define LS3_IRT_ENTRY_HT1(node, x)	LS3_IRT_ENTRY((node), 0x18+(x))

#define LS3_IRT_ROUTE(core, intr)	((0x01 << (core)) | (0x10 << (intr)))

#define LS3_IRQ_INT(x)			(x)		/* sys int 0-3 */
#define LS3_IRQ_PCI(x)			((x) + 0x04)	/* PCI int 0-3 */
#define LS3_IRQ_LPC			0x0a		/* LPC int */
#define LS3_IRQ_HT0(x)			((x) + 0x10)	/* HT0 int 0-7 */
#define LS3_IRQ_HT1(x)			((x) + 0x18)	/* HT1 int 0-7 */
#define LS3_IRQ_NUM			32

#define LS3_IRQ_IS_HT(irq)		((irq) >= 0x10)

#define LS3_IRQ_HT_MASK			0xffff0000u

/*
 * Number of HyperTransport interrupt vectors. In reality, each HT interface
 * has 256 vectors, but the interrupt code uses only a subset of them.
 */
#define LS3_HT_IRQ_NUM			32

/*
 * HyperTransport registers
 */

#define LS3_HT1_MEM_BASE(n)		(LS3_MEM_BASE(n)+0x00000e0000000000ull)
#define LS3_HT1_CFG_BASE(n)		(LS3_MEM_BASE(n)+0x00000efdfb000000ull)

#define LS3_HT_ISR_OFFSET(x)		(0x80 + (x) * 4)
#define LS3_HT_IMR_OFFSET(x)		(0xa0 + (x) * 4)

#endif	/* _MIPS64_LOONGSON3_H_ */
