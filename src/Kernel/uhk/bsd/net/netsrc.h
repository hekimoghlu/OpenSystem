/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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
#ifndef __NET_NETSRC_H__

#include <netinet/in.h>

#define NETSRC_CTLNAME  "com.apple.netsrc"

#define NETSRC_VERSION1 1
#define NETSRC_VERSION2 2
#define NETSRC_CURVERS  NETSRC_VERSION2

struct netsrc_req {
	unsigned int nrq_ver;
	unsigned int nrq_ifscope;
	union {
		union sockaddr_in_4_6 nrq_dst;
		union sockaddr_in_4_6 _usa;
	};
};

#define nrq_sin         _usa.sin
#define nrq_sin6        _usa.sin6

struct netsrc_repv1 {
	union {
		union sockaddr_in_4_6 nrp_src;
		union sockaddr_in_4_6 _usa;
	};
#define NETSRC_IP6_FLAG_TENTATIVE       0x0001
#define NETSRC_IP6_FLAG_TEMPORARY       0x0002
#define NETSRC_IP6_FLAG_DEPRECATED      0x0004
#define NETSRC_IP6_FLAG_OPTIMISTIC      0x0008
#define NETSRC_IP6_FLAG_SECURED         0x0010
	uint16_t nrp_flags;
	uint16_t nrp_label;
	uint16_t nrp_precedence;
	uint16_t nrp_dstlabel;
	uint16_t nrp_dstprecedence;
	uint16_t nrp_unused;    // Padding
};

struct netsrc_repv2 {
	union {
		union sockaddr_in_4_6 nrp_src;
		union sockaddr_in_4_6 _usa;
	};
	uint32_t nrp_min_rtt;
	uint32_t nrp_connection_attempts;
	uint32_t nrp_connection_successes;
	// Continues from above, fixes naming
#define NETSRC_FLAG_IP6_TENTATIVE       NETSRC_IP6_FLAG_TENTATIVE
#define NETSRC_FLAG_IP6_TEMPORARY       NETSRC_IP6_FLAG_TEMPORARY
#define NETSRC_FLAG_IP6_DEPRECATED      NETSRC_IP6_FLAG_DEPRECATED
#define NETSRC_FLAG_IP6_OPTIMISTIC      NETSRC_IP6_FLAG_OPTIMISTIC
#define NETSRC_FLAG_IP6_SECURED         NETSRC_IP6_FLAG_SECURED
#define NETSRC_FLAG_ROUTEABLE           0x00000020
#define NETSRC_FLAG_DIRECT                      0x00000040
#define NETSRC_FLAG_AWDL                        0x00000080
#define NETSRC_FLAG_IP6_DYNAMIC         0x00000100
#define NETSRC_FLAG_IP6_AUTOCONF        0x00000200
	uint32_t nrp_flags;
	uint16_t nrp_label;
	uint16_t nrp_precedence;
	uint16_t nrp_dstlabel;
	uint16_t nrp_dstprecedence;
	uint16_t nrp_ifindex;
	uint16_t nrp_unused; // Padding
};

#define netsrc_rep netsrc_repv2

#define nrp_sin         nrp_src.sin
#define nrp_sin6        nrp_src.sin6

#ifdef KERNEL_PRIVATE
__private_extern__ void netsrc_init(void);
#endif

#endif /* __NET_NETSRC_H__ */
