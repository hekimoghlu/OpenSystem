/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
 * Copyright (c) 2008 The DragonFly Project.  All rights reserved.
 *
 * This code is derived from software contributed to The DragonFly Project
 * by Matthew Dillon <dillon@backplane.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name of The DragonFly Project nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific, prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
 * COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * $DragonFly: src/sys/net/altq/altq_fairq.h,v 1.1 2008/04/06 18:58:15 dillon Exp $
 */

#ifndef _NET_PKTSCHED_PKTSCHED_FAIRQ_H_
#define _NET_PKTSCHED_PKTSCHED_FAIRQ_H_

#ifdef PRIVATE
#include <net/pktsched/pktsched.h>
#include <net/pktsched/pktsched_rmclass.h>
#include <net/classq/classq.h>
#include <net/classq/classq_red.h>
#include <net/classq/classq_rio.h>
#include <net/classq/classq_blue.h>
#include <net/classq/classq_sfb.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FAIRQ_MAX_BUCKETS       2048    /* maximum number of sorting buckets */
#define FAIRQ_MAXPRI            RM_MAXPRIO
#define FAIRQ_BITMAP_WIDTH      (sizeof (fairq_bitmap_t) * 8)
#define FAIRQ_BITMAP_MASK       (FAIRQ_BITMAP_WIDTH - 1)

/* fairq class flags */
#define FARF_RED                0x0001  /* use RED */
#define FARF_ECN                0x0002  /* use ECN with RED/BLUE/SFB */
#define FARF_RIO                0x0004  /* use RIO */
#define FARF_CLEARDSCP          0x0010  /* clear diffserv codepoint */
#define FARF_BLUE               0x0100  /* use BLUE */
#define FARF_SFB                0x0200  /* use SFB */
#define FARF_FLOWCTL            0x0400  /* enable flow control advisories */
#define FARF_DEFAULTCLASS       0x1000  /* default class */
#ifdef BSD_KERNEL_PRIVATE
#define FARF_HAS_PACKETS        0x2000  /* might have queued packets */
#define FARF_LAZY               0x10000000 /* on-demand resource allocation */
#endif /* BSD_KERNEL_PRIVATE */

#define FARF_USERFLAGS                                                  \
	(FARF_RED | FARF_ECN | FARF_RIO | FARF_CLEARDSCP |              \
	FARF_BLUE | FARF_SFB | FARF_FLOWCTL | FARF_DEFAULTCLASS)

#ifdef BSD_KERNEL_PRIVATE
#define FARF_BITS \
	"\020\1RED\2ECN\3RIO\5CLEARDSCP\11BLUE\12SFB\13FLOWCTL\15DEFAULT" \
	"\16HASPKTS\35LAZY"
#else
#define FARF_BITS \
	"\020\1RED\2ECN\3RIO\5CLEARDSCP\11BLUE\12SFB\13FLOWCTL\15DEFAULT" \
	"\16HASPKTS"
#endif /* !BSD_KERNEL_PRIVATE */

typedef u_int32_t       fairq_bitmap_t;

struct fairq_classstats {
	u_int32_t               class_handle;
	u_int32_t               priority;

	u_int32_t               qlength;
	u_int32_t               qlimit;
	struct pktcntr          xmit_cnt;  /* transmitted packet counter */
	struct pktcntr          drop_cnt;  /* dropped packet counter */

	/* RED, RIO, BLUE, SFB related info */
	classq_type_t           qtype;
	union {
		/* RIO has 3 red stats */
		struct red_stats        red[RIO_NDROPPREC];
		struct blue_stats       blue;
		struct sfb_stats        sfb;
	};
	classq_state_t          qstate;
};

#ifdef __cplusplus
}
#endif
#endif /* PRIVATE */
#endif /* _NET_PKTSCHED_PKTSCHED_FAIRQ_H_ */
