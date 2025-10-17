/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
 * Copyright (c) 2001 Daniel Hartmeier
 * Copyright (c) 2002 - 2013 Henning Brauer
 * NAT64 - Copyright (c) 2010 Viagenie Inc. (http://www.viagenie.ca)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *    - Redistributions of source code must retain the above copyright
 *	notice, this list of conditions and the following disclaimer.
 *    - Redistributions in binary form must reproduce the above
 *	copyright notice, this list of conditions and the following
 *	disclaimer in the documentation and/or other materials provided
 *	with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Effort sponsored in part by the Defense Advanced Research Projects
 * Agency (DARPA) and Air Force Research Laboratory, Air Force
 * Materiel Command, USAF, under agreement number F30602-01-2-0537.
 *
 */
#ifndef _NET_NAT464_UTILS_H_
#define _NET_NAT464_UTILS_H_
#include <netinet/in.h>
#include <net/pf_pbuf.h>

#define clat_log0(x)    do { log x; } while (0)
#define clat_log1(x)    do { if (clat_debug >= 1) log x; } while (0)
#define clat_log2(x)    do { if (clat_debug >= 2) log x; } while (0)

#define CLAT46_NEEDED(x)                                                        \
	(!IN_LOOPBACK(x) && !IN_LINKLOCAL(x) && !IN_MULTICAST(x) &&             \
	INADDR_BROADCAST != x)

#define CLAT64_NEEDED(x)                                                        \
	(!IN6_IS_ADDR_LOOPBACK(x) && !IN6_IS_ADDR_LINKLOCAL(x) &&               \
	!IN6_IS_ADDR_MULTICAST(x))

extern int clat_debug;

enum    { NT_DROP, NT_NAT64 };
enum    { NT_IN, NT_OUT };
struct nat464_addr {
	union {
		struct in_addr          _v4addr;
		struct in6_addr         _v6addr;
		uint8_t         _addr8[16];
		uint16_t                _addr16[8];
		uint32_t                _addr32[4];
	} nat464a;                  /* 128-bit address */
#define natv4addr       nat464a._v4addr
#define natv6addr       nat464a._v6addr
#define nataddr8        nat464a._addr8
#define nataddr16       nat464a._addr16
#define nataddr32       nat464a._addr32
};

int
nat464_translate_icmp(int, void *);

int
    nat464_translate_icmp_ip(pbuf_t *, uint16_t, uint16_t *, uint16_t *,
    uint8_t, uint8_t, uint16_t, struct nat464_addr *,
    struct nat464_addr *, protocol_family_t, protocol_family_t );

int
    nat464_synthesize_ipv6(ifnet_t, const struct in_addr *, struct in6_addr *);

int
    nat464_synthesize_ipv4(ifnet_t, const struct in6_addr *, struct in_addr *);

int
    nat464_translate_64(pbuf_t *, int, uint8_t, uint8_t *, uint8_t, struct in_addr,
    struct in_addr, uint64_t, boolean_t *);

int
    nat464_translate_46(pbuf_t *, uint16_t, uint8_t, uint8_t, uint8_t, struct in6_addr,
    struct in6_addr, uint16_t);

int
    nat464_translate_proto(pbuf_t *, struct nat464_addr *, struct nat464_addr *,
    uint8_t, protocol_family_t, protocol_family_t, int, boolean_t);

int
    nat464_insert_frag46(pbuf_t *, uint16_t, uint16_t, boolean_t);

int
    nat464_remove_frag64(pbuf_t *, uint32_t, uint16_t, boolean_t);

uint16_t
    nat464_cksum_fixup(uint16_t, uint16_t, uint16_t, uint8_t);
#endif /* !_NET_NAT464_UTILS_H_ */
