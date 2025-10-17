/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
/* \summary: IPv6 fragmentation header printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"
#include "extract.h"

#include "ip6.h"

int
frag6_print(netdissect_options *ndo, const u_char *bp, const u_char *bp2)
{
	const struct ip6_frag *dp;
	const struct ip6_hdr *ip6;

	ndo->ndo_protocol = "frag6";
	dp = (const struct ip6_frag *)bp;
	ip6 = (const struct ip6_hdr *)bp2;

	if (ndo->ndo_vflag) {
		ND_PRINT("frag (0x%08x:%u|%zu)",
			 GET_BE_U_4(dp->ip6f_ident),
			 GET_BE_U_2(dp->ip6f_offlg) & IP6F_OFF_MASK,
			 sizeof(struct ip6_hdr) + GET_BE_U_2(ip6->ip6_plen) -
			        (bp - bp2) - sizeof(struct ip6_frag));
	} else {
		ND_PRINT("frag (%u|%zu)",
		         GET_BE_U_2(dp->ip6f_offlg) & IP6F_OFF_MASK,
		         sizeof(struct ip6_hdr) + GET_BE_U_2(ip6->ip6_plen) -
			         (bp - bp2) - sizeof(struct ip6_frag));
	}

	/* it is meaningless to decode non-first fragment */
	if ((GET_BE_U_2(dp->ip6f_offlg) & IP6F_OFF_MASK) != 0)
		return -1;
	else
	{
		ND_PRINT(" ");
		return sizeof(struct ip6_frag);
	}
}
