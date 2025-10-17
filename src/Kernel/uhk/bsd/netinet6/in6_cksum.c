/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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

/*
 * Copyright (c) 1988, 1992, 1993
 *	The Regents of the University of California.  All rights reserved.
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
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
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
 *
 *	@(#)in_cksum.c	8.1 (Berkeley) 6/10/93
 */

/*-
 * Copyright (c) 2008 Joerg Sonnenberger <joerg@NetBSD.org>.
 * All rights reserved.
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
 */

#include <sys/param.h>
#include <machine/endian.h>
#include <sys/mbuf.h>
#include <sys/systm.h>
#include <kern/debug.h>
#include <netinet/in.h>
#include <netinet/ip6.h>
#include <netinet6/ip6_var.h>

/*
 * Checksum routine for Internet Protocol family headers (Portable Version).
 *
 * This routine is very heavily used in the network
 * code and should be modified for each CPU to be as fast as possible.
 */

/*
 * Compute IPv6 pseudo-header checksum; assumes 16-bit aligned pointers.
 */
uint16_t
in6_pseudo(const struct in6_addr *src, const struct in6_addr *dst, uint32_t x)
{
	uint32_t sum = 0;
	const uint16_t *w;

	/*
	 * IPv6 source address
	 */
	w = (const uint16_t *)(const struct in6_addr *__bidi_indexable)src;
	sum += w[0];
	if (!IN6_IS_SCOPE_EMBED(src)) {
		sum += w[1];
	}
	sum += w[2]; sum += w[3]; sum += w[4]; sum += w[5];
	sum += w[6]; sum += w[7];

	/*
	 * IPv6 destination address
	 */
	w = (const uint16_t *)(const struct in6_addr *__bidi_indexable)dst;
	sum += w[0];
	if (!IN6_IS_SCOPE_EMBED(dst)) {
		sum += w[1];
	}
	sum += w[2]; sum += w[3]; sum += w[4]; sum += w[5];
	sum += w[6]; sum += w[7];

	/*
	 * Caller-supplied value; 'x' could be one of:
	 *
	 *	htonl(proto + length), or
	 *	htonl(proto + length + sum)
	 **/
	sum += x;

	/* fold in carry bits */
	ADDCARRY(sum);

	return (uint16_t)sum;
}

/*
 * m MUST contain at least an IPv6 header, if nxt is specified;
 * nxt is the upper layer protocol number;
 * off is an offset where TCP/UDP/ICMP6 header starts;
 * len is a total length of a transport segment (e.g. TCP header + TCP payload)
 */
u_int16_t
inet6_cksum(struct mbuf *m, uint32_t nxt, uint32_t off, uint32_t len)
{
	uint32_t sum;

	sum = m_sum16(m, off, len);

	if (nxt != 0) {
		struct ip6_hdr *__single ip6;
		unsigned char buf[sizeof(*ip6)] __attribute__((aligned(8)));
		uint32_t mlen;

		/*
		 * Sanity check
		 *
		 * Use m_length2() instead of m_length(), as we cannot rely on
		 * the caller setting m_pkthdr.len correctly, if the mbuf is
		 * a M_PKTHDR one.
		 */
		if ((mlen = m_length2(m, NULL)) < sizeof(*ip6)) {
			panic("%s: mbuf %p pkt too short (%d) for IPv6 header",
			    __func__, m, mlen);
			/* NOTREACHED */
		}

		/*
		 * In case the IPv6 header is not contiguous, or not 32-bit
		 * aligned, copy it to a local buffer.  Note here that we
		 * expect the data pointer to point to the IPv6 header.
		 */
		if ((sizeof(*ip6) > m->m_len) ||
		    !IP6_HDR_ALIGNED_P(mtod(m, caddr_t))) {
			m_copydata(m, 0, sizeof(*ip6), (caddr_t)buf);
			ip6 = (struct ip6_hdr *)(void *)buf;
		} else {
			ip6 = (struct ip6_hdr *)m_mtod_current(m);
		}

		/* add pseudo header checksum */
		sum += in6_pseudo(&ip6->ip6_src, &ip6->ip6_dst,
		    htonl(nxt + len));

		/* fold in carry bits */
		ADDCARRY(sum);
	}

	return ~sum & 0xffff;
}

/*
 * buffer MUST contain at least an IPv6 header, if nxt is specified;
 * nxt is the upper layer protocol number;
 * off is an offset where TCP/UDP/ICMP6 header starts;
 * len is a total length of a transport segment (e.g. TCP header + TCP payload)
 */
u_int16_t
inet6_cksum_buffer(const uint8_t *__sized_by(buffer_len)buffer, uint32_t nxt, uint32_t off,
    uint32_t len, uint32_t buffer_len)
{
	uint32_t sum;
	uint32_t tmp;

	if (__improbable(off >= buffer_len)) {
		panic("%s: off (%u) >=  buffer_len (%u)",
		    __func__, off, buffer_len);
		/* NOTREACHED */
	}

	if (__improbable(len == 0 || len > buffer_len)) {
		panic("%s: len == 0 OR len (%u) >=  buffer_len (%u)",
		    __func__, len, buffer_len);
		/* NOTREACHED */
	}

	if (__improbable(os_add_overflow(off, len, &tmp))) {
		panic("%s: off(%u), len(%u) add overflow",
		    __func__, off, len);
		/* NOTREACHED */
	}

	if (__improbable(tmp > buffer_len)) {
		panic("%s: off(%u) + len(%u) >=  buffer_len (%u)",
		    __func__, off, len, buffer_len);
		/* NOTREACHED */
	}

	sum = b_sum16(&((const uint8_t *)buffer)[off], len);

	if (nxt != 0) {
		const struct ip6_hdr *ip6;
		unsigned char buf[sizeof(*ip6)] __attribute__((aligned(8)));

		/*
		 * In case the IPv6 header is not contiguous, or not 32-bit
		 * aligned, copy it to a local buffer.  Note here that we
		 * expect the data pointer to point to the IPv6 header.
		 */
		if (!IP6_HDR_ALIGNED_P(buffer)) {
			memcpy(buf, buffer, sizeof(*ip6));
			ip6 = (const struct ip6_hdr *)(const void *)buf;
		} else {
			ip6 = (const struct ip6_hdr *)buffer;
		}

		/* add pseudo header checksum */
		sum += in6_pseudo(&ip6->ip6_src, &ip6->ip6_dst,
		    htonl(nxt + len));

		/* fold in carry bits */
		ADDCARRY(sum);
	}

	return ~sum & 0xffff;
}
