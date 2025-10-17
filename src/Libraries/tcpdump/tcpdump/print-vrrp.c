/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 24, 2023.
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
/* \summary: Virtual Router Redundancy Protocol (VRRP) printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"
#include "extract.h"
#include "addrtoname.h"

#include "ip.h"
#include "ipproto.h"
/*
 * RFC 2338 (VRRP v2):
 *
 *     0                   1                   2                   3
 *     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |Version| Type  | Virtual Rtr ID|   Priority    | Count IP Addrs|
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |   Auth Type   |   Adver Int   |          Checksum             |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |                         IP Address (1)                        |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |                            .                                  |
 *    |                            .                                  |
 *    |                            .                                  |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |                         IP Address (n)                        |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |                     Authentication Data (1)                   |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |                     Authentication Data (2)                   |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *
 *
 * RFC 5798 (VRRP v3):
 *
 *    0                   1                   2                   3
 *    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |                    IPv4 Fields or IPv6 Fields                 |
 *   ...                                                             ...
 *    |                                                               |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |Version| Type  | Virtual Rtr ID|   Priority    |Count IPvX Addr|
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |(rsvd) |     Max Adver Int     |          Checksum             |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |                                                               |
 *    +                                                               +
 *    |                       IPvX Address(es)                        |
 *    +                                                               +
 *    |                                                               |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */

/* Type */
#define	VRRP_TYPE_ADVERTISEMENT	1

static const struct tok type2str[] = {
	{ VRRP_TYPE_ADVERTISEMENT,	"Advertisement"	},
	{ 0,				NULL		}
};

/* Auth Type */
#define	VRRP_AUTH_NONE		0
#define	VRRP_AUTH_SIMPLE	1
#define	VRRP_AUTH_AH		2

static const struct tok auth2str[] = {
	{ VRRP_AUTH_NONE,		"none"		},
	{ VRRP_AUTH_SIMPLE,		"simple"	},
	{ VRRP_AUTH_AH,			"ah"		},
	{ 0,				NULL		}
};

void
vrrp_print(netdissect_options *ndo,
           const u_char *bp, u_int len,
           const u_char *bp2, int ttl)
{
	int version, type, auth_type = VRRP_AUTH_NONE; /* keep compiler happy */
	const char *type_s;

	ndo->ndo_protocol = "vrrp";
	version = (GET_U_1(bp) & 0xf0) >> 4;
	type = GET_U_1(bp) & 0x0f;
	type_s = tok2str(type2str, "unknown type (%u)", type);
	ND_PRINT("VRRPv%u, %s", version, type_s);
	if (ttl != 255)
		ND_PRINT(", (ttl %u)", ttl);
	if (version < 2 || version > 3 || type != VRRP_TYPE_ADVERTISEMENT)
		return;
	ND_PRINT(", vrid %u, prio %u", GET_U_1(bp + 1), GET_U_1(bp + 2));

	if (version == 2) {
		auth_type = GET_U_1(bp + 4);
		ND_PRINT(", authtype %s", tok2str(auth2str, NULL, auth_type));
		ND_PRINT(", intvl %us, length %u", GET_U_1(bp + 5), len);
	} else { /* version == 3 */
		uint16_t intvl = (GET_U_1(bp + 4) & 0x0f) << 8 | GET_U_1(bp + 5);
		ND_PRINT(", intvl %ucs, length %u", intvl, len);
	}

	if (ndo->ndo_vflag) {
		u_int naddrs = GET_U_1(bp + 3);
		u_int i;
		char c;

		if (version == 2 && ND_TTEST_LEN(bp, len)) {
			struct cksum_vec vec[1];

			vec[0].ptr = bp;
			vec[0].len = len;
			if (in_cksum(vec, 1))
				ND_PRINT(", (bad vrrp cksum %x)",
					GET_BE_U_2(bp + 6));
		}

		if (version == 3 && ND_TTEST_LEN(bp, len)) {
			uint16_t cksum = nextproto4_cksum(ndo, (const struct ip *)bp2, bp,
				len, len, IPPROTO_VRRP);
			if (cksum)
				ND_PRINT(", (bad vrrp cksum %x)",
					GET_BE_U_2(bp + 6));
		}

		ND_PRINT(", addrs");
		if (naddrs > 1)
			ND_PRINT("(%u)", naddrs);
		ND_PRINT(":");
		c = ' ';
		bp += 8;
		for (i = 0; i < naddrs; i++) {
			ND_PRINT("%c%s", c, GET_IPADDR_STRING(bp));
			c = ',';
			bp += 4;
		}
		if (version == 2 && auth_type == VRRP_AUTH_SIMPLE) { /* simple text password */
			ND_PRINT(" auth \"");
			/*
			 * RFC 2338 Section 5.3.10: "If the configured authentication string
			 * is shorter than 8 bytes, the remaining space MUST be zero-filled.
			 */
			nd_printjnp(ndo, bp, 8);
			ND_PRINT("\"");
		}
	}
}
