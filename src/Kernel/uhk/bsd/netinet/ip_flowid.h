/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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
 * Copyright (c) 2002 Luigi Rizzo, Universita` di Pisa
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

#ifndef __IP_FLOWID_H__
#define __IP_FLOWID_H__

#include <sys/types.h>
#include <netinet/in.h>

/*
 * This structure is used as a flow mask and a flow id for various
 * parts of the code.
 */
struct ip_flow_id {
	u_int32_t       dst_ip;
	u_int32_t       src_ip;
	u_int16_t       dst_port;
	u_int16_t       src_port;
	u_int8_t        proto;
	u_int8_t        flags;  /* protocol-specific flags */
	u_int8_t        addr_type; /* 4 = ipv4, 6 = ipv6, 1=ether ? */
	struct in6_addr dst_ip6;        /* could also store MAC addr! */
	struct in6_addr src_ip6;
	u_int32_t       flow_id6;
	u_int32_t       frag_id6;
};

#define IS_IP6_FLOW_ID(id)      ((id)->addr_type == 6)

#ifdef BSD_KERNEL_PRIVATE
struct route_in6;
struct sockaddr_in6;
struct pf_rule;

/*
 * Arguments for calling ipfw_chk() and dummynet_io(). We put them
 * all into a structure because this way it is easier and more
 * efficient to pass variables around and extend the interface.
 */
struct ip_fw_args {
	struct mbuf             *fwa_m;         /* the mbuf chain               */
	struct ifnet            *fwa_oif;       /* output interface             */
	struct pf_rule          *fwa_pf_rule;   /* matching PF rule             */
	struct ether_header     *fwa_eh;        /* for bridged packets          */
	int                     fwa_flags;      /* for dummynet                 */
	int                     fwa_oflags;     /* for dummynet         */
	union {
		struct ip_out_args  *_fwa_ipoa;     /* for dummynet                */
		struct ip6_out_args *_fwa_ip6oa;    /* for dummynet               */
	} fwa_ipoa_;
	union {
		struct route        *_fwa_ro;       /* for dummynet         */
		struct route_in6    *_fwa_ro6;      /* for dummynet         */
	} fwa_ro_;
	union {
		struct sockaddr_in  *_fwa_dst;      /* for dummynet         */
		struct sockaddr_in6 *_fwa_dst6;     /* for IPv6 dummynet         */
	} fwa_dst_;
	struct route_in6        *fwa_ro6_pmtu;  /* for IPv6 output */
	struct ifnet            *fwa_origifp;   /* for IPv6 output */
	u_int32_t               fwa_mtu;        /* for IPv6 output */
	u_int32_t               fwa_unfragpartlen;  /* for IPv6 output */
	struct ip6_exthdrs      *fwa_exthdrs;   /* for IPv6 output */
	struct ip_flow_id       fwa_id;         /* grabbed from IP header       */
	u_int32_t               fwa_cookie;
};
#define fwa_ipoa fwa_ipoa_._fwa_ipoa
#define fwa_ip6oa fwa_ipoa_._fwa_ip6oa
#define fwa_ro fwa_ro_._fwa_ro
#define fwa_ro6 fwa_ro_._fwa_ro6
#define fwa_dst fwa_dst_._fwa_dst
#define fwa_dst6 fwa_dst_._fwa_dst6

/* Allocate a separate structure for inputs args to save space and bzero time */
struct ip_fw_in_args {
	struct pf_rule          *fwai_pf_rule;  /* matching PF rule           */
};

#endif /* BSD_KERNEL_PRIVATE */

#endif /* __IP_FLOWID_H__ */
