/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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

#include <sys/param.h>
#include <sys/mcache.h>
#include <sys/mbuf.h>
#include <sys/socket.h>
#include <sys/queue.h>
#include <kern/debug.h>
#include <string.h>

#include <net/if.h>
#include <net/route.h>

#include <netinet/in.h>
#include <netinet6/in6_var.h>
#include <netinet/ip6.h>
#include <netinet6/ip6_var.h>

#include <netinet/icmp6.h>

int
route6_input(struct mbuf **mp, int *offp, int proto)
{
#pragma unused(proto)
	struct ip6_hdr *ip6 = NULL;
	mbuf_ref_t m = *mp;
	struct ip6_rthdr *__single rh = NULL;
	int off = *offp, rhlen = 0;
#ifdef notyet
	struct ip6aux *__single ip6a = NULL;

	ip6a = ip6_findaux(m);
	if (ip6a) {
		/* XXX reject home-address option before rthdr */
		if (ip6a->ip6a_flags & IP6A_SWAP) {
			ip6stat.ip6s_badoptions++;
			*mp = NULL;
			m_freem(m);
			return IPPROTO_DONE;
		}
	}
	ip6a = NULL;
#endif /* notyet */

	IP6_EXTHDR_CHECK(m, off, sizeof(*rh), return IPPROTO_DONE);

	/* Expect 32-bit aligned data pointer on strict-align platforms */
	MBUF_STRICT_DATA_ALIGNMENT_CHECK_32(m);

	ip6 = mtod(m, struct ip6_hdr *);
	rh = (struct ip6_rthdr *)((caddr_t)ip6 + off);

	switch (rh->ip6r_type) {
	default:
		/* unknown routing type */
		if (rh->ip6r_segleft == 0) {
			rhlen = (rh->ip6r_len + 1) << 3;
			break;  /* Final dst. Just ignore the header. */
		}
		ip6stat.ip6s_badoptions++;
		icmp6_error(m, ICMP6_PARAM_PROB, ICMP6_PARAMPROB_HEADER,
		    (int)((caddr_t)&rh->ip6r_type - (caddr_t)ip6));
		return IPPROTO_DONE;
	}

	*mp = m;
	*offp += rhlen;
	return rh->ip6r_nxt;
}
