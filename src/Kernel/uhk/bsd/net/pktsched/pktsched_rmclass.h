/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
/* $OpenBSD: altq_rmclass.h,v 1.10 2007/06/17 19:58:58 jasper Exp $	*/
/* $KAME: altq_rmclass.h,v 1.6 2000/12/09 09:22:44 kjc Exp $	*/

/*
 * Copyright (c) 1991-1997 Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the Network Research
 *	Group at Lawrence Berkeley Laboratory.
 * 4. Neither the name of the University nor of the Laboratory may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _NET_PKTSCHED_PKTSCHED_RMCLASS_H_
#define _NET_PKTSCHED_PKTSCHED_RMCLASS_H_

#ifdef PRIVATE
#include <net/classq/classq.h>
#include <net/pktsched/pktsched.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RM_MAXPRIO      8       /* Max priority */

/* flags for rmc_init and rmc_newclass */
/* class flags */
#define RMCF_RED                0x0001  /* use RED */
#define RMCF_ECN                0x0002  /* use ECN with RED/BLUE/SFB */
#define RMCF_RIO                0x0004  /* use RIO */
#define RMCF_FLOWVALVE          0x0008  /* use flowvalve (aka penalty-box) */
#define RMCF_CLEARDSCP          0x0010  /* clear diffserv codepoint */

/* flags for rmc_init */
#define RMCF_WRR                0x0100
#define RMCF_EFFICIENT          0x0200

#define RMCF_BLUE               0x10000 /* use BLUE */
#define RMCF_SFB                0x20000 /* use SFB */
#define RMCF_FLOWCTL            0x40000 /* enable flow control advisories */

#ifdef __cplusplus
}
#endif
#endif /* PRIVATE */
#endif /* _NET_PKTSCHED_PKTSCHED_RMCLASS_H_ */
