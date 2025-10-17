/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
/* \summary: Loopback Protocol printer */

/*
 * originally defined as the Ethernet Configuration Testing Protocol.
 * specification: https://www.mit.edu/people/jhawk/ctp.pdf
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "extract.h"
#include "addrtoname.h"


#define LOOPBACK_REPLY   1
#define LOOPBACK_FWDDATA 2

static const struct tok fcode_str[] = {
	{ LOOPBACK_REPLY,   "Reply"        },
	{ LOOPBACK_FWDDATA, "Forward Data" },
	{ 0, NULL }
};

static void
loopback_message_print(netdissect_options *ndo,
                       const u_char *cp, u_int len)
{
	uint16_t function;

	if (len < 2)
		goto invalid;
	/* function */
	function = GET_LE_U_2(cp);
	cp += 2;
	len -= 2;
	ND_PRINT(", %s", tok2str(fcode_str, " invalid (%u)", function));

	switch (function) {
		case LOOPBACK_REPLY:
			if (len < 2)
				goto invalid;
			/* receipt number */
			ND_PRINT(", receipt number %u", GET_LE_U_2(cp));
			cp += 2;
			len -= 2;
			/* data */
			ND_PRINT(", data (%u octets)", len);
			ND_TCHECK_LEN(cp, len);
			break;
		case LOOPBACK_FWDDATA:
			if (len < MAC_ADDR_LEN)
				goto invalid;
			/* forwarding address */
			ND_PRINT(", forwarding address %s", GET_ETHERADDR_STRING(cp));
			cp += MAC_ADDR_LEN;
			len -= MAC_ADDR_LEN;
			/* data */
			ND_PRINT(", data (%u octets)", len);
			ND_TCHECK_LEN(cp, len);
			break;
		default:
			ND_TCHECK_LEN(cp, len);
			break;
	}
	return;

invalid:
	nd_print_invalid(ndo);
	ND_TCHECK_LEN(cp, len);
}

void
loopback_print(netdissect_options *ndo,
               const u_char *cp, u_int len)
{
	uint16_t skipCount;

	ndo->ndo_protocol = "loopback";
	ND_PRINT("Loopback");
	if (len < 2)
		goto invalid;
	/* skipCount */
	skipCount = GET_LE_U_2(cp);
	cp += 2;
	len -= 2;
	ND_PRINT(", skipCount %u", skipCount);
	if (skipCount % 8)
		ND_PRINT(" (bogus)");
	if (skipCount > len)
		goto invalid;
	/* the octets to skip */
	ND_TCHECK_LEN(cp, skipCount);
	cp += skipCount;
	len -= skipCount;
	/* the first message to decode */
	loopback_message_print(ndo, cp, len);
	return;

invalid:
	nd_print_invalid(ndo);
	ND_TCHECK_LEN(cp, len);
}

