/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 19, 2024.
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
 * Copyright (c) 1982, 1986, 1993
 *      The Regents of the University of California.  All rights reserved.
 */
#ifndef __KDP_UDP_H
#define __KDP_UDP_H

#include <libsa/types.h>
#include <libkern/OSByteOrder.h>   /* OSSwap functions */
#include <stdint.h>

struct kdp_in_addr {
	uint32_t s_addr;
};

#define ETHER_ADDR_LEN 6

struct kdp_ether_addr {
	u_char ether_addr_octet[ETHER_ADDR_LEN];
};

typedef struct kdp_ether_addr enet_addr_t;

extern struct kdp_ether_addr kdp_get_mac_addr(void);
unsigned int  kdp_get_ip_address(void);

struct  kdp_ether_header {
	u_char  ether_dhost[ETHER_ADDR_LEN];
	u_char  ether_shost[ETHER_ADDR_LEN];
	u_short ether_type;
};

typedef struct kdp_ether_header ether_header_t;

#define ntohs(x)           OSSwapBigToHostInt16(x)
#define ntohl(x)           OSSwapBigToHostInt32(x)
#define htons(x)           OSSwapHostToBigInt16(x)
#define htonl(x)           OSSwapHostToBigInt32(x)

/*
 * IONetworkingFamily only.
 */
typedef uint32_t (*kdp_link_t)(void);
typedef boolean_t (*kdp_mode_t)(boolean_t);
void    kdp_register_link(kdp_link_t link, kdp_mode_t mode);
void    kdp_unregister_link(kdp_link_t link, kdp_mode_t mode);

#endif /* __KDP_UDP_H */
