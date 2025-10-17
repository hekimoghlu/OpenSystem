/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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
/* \summary: IP Payload Compression Protocol (IPComp) printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"
#include "extract.h"

struct ipcomp {
	nd_uint8_t  comp_nxt;	/* Next Header */
	nd_uint8_t  comp_flags;	/* Length of data, in 32bit */
	nd_uint16_t comp_cpi;	/* Compression parameter index */
};

void
ipcomp_print(netdissect_options *ndo, const u_char *bp)
{
	const struct ipcomp *ipcomp;
	uint16_t cpi;

	ndo->ndo_protocol = "ipcomp";
	ipcomp = (const struct ipcomp *)bp;
	cpi = GET_BE_U_2(ipcomp->comp_cpi);

	ND_PRINT("IPComp(cpi=0x%04x)", cpi);

	/*
	 * XXX - based on the CPI, we could decompress the packet here.
	 * Packet buffer management is a headache (if we decompress,
	 * packet will become larger).
	 *
	 * We would decompress the packet and then call a routine that,
	 * based on ipcomp->comp_nxt, dissects the decompressed data.
	 *
	 * Until we do that, however, we just return -1, so that
	 * the loop that processes "protocol"/"next header" types
	 * stops - there's nothing more it can do with a compressed
	 * payload.
	 */
}
