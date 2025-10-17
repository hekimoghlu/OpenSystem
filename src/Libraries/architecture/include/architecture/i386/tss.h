/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 1, 2024.
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
 * Copyright (c) 1992 NeXT Computer, Inc.
 *
 * Intel386 Family:	Task State Segment.
 *
 * HISTORY
 *
 * 29 March 1992 ? at NeXT
 *	Created.
 */

#ifndef _ARCH_I386_TSS_H_
#define _ARCH_I386_TSS_H_

#include <architecture/i386/sel.h>

/*
 * Task State segment.
 */

typedef struct tss {
    sel_t		oldtss;
    unsigned int		:0;
    unsigned int	esp0;
    sel_t		ss0;
    unsigned int		:0;
    unsigned int	esp1;
    sel_t		ss1;
    unsigned int		:0;
    unsigned int	esp2;
    sel_t		ss2;
    unsigned int		:0;
    unsigned int	cr3;
    unsigned int	eip;
    unsigned int	eflags;
    unsigned int	eax;
    unsigned int	ecx;
    unsigned int	edx;
    unsigned int	ebx;
    unsigned int	esp;
    unsigned int	ebp;
    unsigned int	esi;
    unsigned int	edi;
    sel_t		es;
    unsigned int		:0;
    sel_t		cs;
    unsigned int		:0;
    sel_t		ss;
    unsigned int		:0;
    sel_t		ds;
    unsigned int		:0;
    sel_t		fs;
    unsigned int		:0;
    sel_t		gs;
    unsigned int		:0;
    sel_t		ldt;
    unsigned int		:0;
    unsigned int	t	:1,
    				:15,
			io_bmap	:16;
} tss_t;

#define TSS_SIZE(n)	(sizeof (struct tss) + (n))

/*
 * Task State segment descriptor.
 */

typedef struct tss_desc {
    unsigned short	limit00;
    unsigned short	base00;
    unsigned char	base16;
    unsigned char	type	:5,
#define DESC_TSS	0x09
			dpl	:2,
			present	:1;
    unsigned char	limit16	:4,
				:3,
			granular:1;
    unsigned char	base24;
} tss_desc_t;

/*
 * Task gate descriptor.
 */

typedef struct task_gate {
    unsigned short		:16;
    sel_t		tss;
    unsigned int		:8,
    			type	:5,
#define DESC_TASK_GATE	0x05
			dpl	:2,
			present	:1,
				:0;
} task_gate_t;

#endif	/* _ARCH_I386_TSS_H_ */
