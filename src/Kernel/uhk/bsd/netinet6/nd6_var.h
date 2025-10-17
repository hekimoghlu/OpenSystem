/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
 * Copyright (C) 1995, 1996, 1997, and 1998 WIDE Project.
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
 * 3. Neither the name of the project nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _NETINET6_ND6_VAR_H_
#define _NETINET6_ND6_VAR_H_

#ifdef BSD_KERNEL_PRIVATE
struct nd_ifinfo {
	decl_lck_mtx_data(, lock);
	boolean_t initialized;          /* Flag to see the entry is initialized */
	u_int32_t linkmtu;              /* LinkMTU */
	u_int32_t maxmtu;               /* Upper bound of LinkMTU */
	u_int32_t basereachable;        /* BaseReachableTime */
	u_int32_t reachable;            /* Reachable Time */
	u_int32_t retrans;              /* Retrans Timer */
	u_int32_t flags;                /* Flags */
	int recalctm;                   /* BaseReacable re-calculation timer */
	u_int8_t chlim;                 /* CurHopLimit */
	u_int8_t _pad[3];
	/* the following 3 members are for privacy extension for addrconf */
	u_int8_t randomseed0[8]; /* upper 64 bits of SHA256 digest */
	u_int8_t randomseed1[8]; /* lower 64 bits (usually the EUI64 IFID) */
	u_int8_t randomid[8];   /* current random ID */
	/* keep track of routers and prefixes on this link */
	int32_t nprefixes;
	int32_t ndefrouters;
	boolean_t cga_initialized;
	struct in6_cga_modifier local_cga_modifier;
	uint8_t cga_collision_count;
};
#endif /* BSD_KERNEL_PRIVATE */
#endif /* _NETINET6_ND6_VAR_H_ */
