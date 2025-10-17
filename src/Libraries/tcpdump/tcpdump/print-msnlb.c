/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
/* \summary: MS Network Load Balancing's (NLB) heartbeat printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"
#include "addrtoname.h"
#include "extract.h"

struct msnlb_heartbeat_pkt {
	nd_byte     unknown1[4];
	nd_byte     unknown2[4];
	nd_uint32_t host_prio;	/* little-endian */
	nd_ipv4     virtual_ip;
	nd_ipv4     host_ip;
	/* the protocol is undocumented so we ignore the rest */
};

void
msnlb_print(netdissect_options *ndo, const u_char *bp)
{
	const struct msnlb_heartbeat_pkt *hb;

	ndo->ndo_protocol = "msnlb";
	hb = (const struct msnlb_heartbeat_pkt *)bp;

	ND_PRINT("MS NLB heartbeat");
	ND_PRINT(", host priority: %u", GET_LE_U_4((hb->host_prio)));
	ND_PRINT(", cluster IP: %s", GET_IPADDR_STRING(hb->virtual_ip));
	ND_PRINT(", host IP: %s", GET_IPADDR_STRING(hb->host_ip));
}
