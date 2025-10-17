/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
/* \summary: Broadcom LI Printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "addrtoname.h"
#include "extract.h"

#define BCM_LI_SHIM_LEN	4

static const struct tok bcm_li_direction_values[] = {
    { 1, "unused" },
    { 2, "egress" },
    { 3, "ingress" },
    { 0, NULL}
};

#define BCM_LI_PKT_TYPE_UNDECIDED 4
#define BCM_LI_PKT_TYPE_IPV4      5
#define BCM_LI_PKT_TYPE_IPV6      6
#define BCM_LI_PKT_TYPE_ETHERNET  7

static const struct tok bcm_li_pkt_type_values[] = {
    { BCM_LI_PKT_TYPE_UNDECIDED, "undecided" },
    { BCM_LI_PKT_TYPE_IPV4, "ipv4" },
    { BCM_LI_PKT_TYPE_IPV6, "ipv6" },
    { BCM_LI_PKT_TYPE_ETHERNET, "ethernet" },
    { 0, NULL}
};

static const struct tok bcm_li_pkt_subtype_values[] = {
    { 1, "single VLAN tag" },
    { 2, "double VLAN tag" },
    { 3, "untagged" },
    { 0, NULL}
};

void
bcm_li_print(netdissect_options *ndo,
             const u_char *bp, u_int length)
{
	u_int shim, direction, pkt_type, pkt_subtype, li_id;

	ndo->ndo_protocol = "bcm_li";
	if (length < BCM_LI_SHIM_LEN) {
	    ND_PRINT(" (length %u < %u)", length, BCM_LI_SHIM_LEN);
	    goto invalid;
	}
	shim = GET_BE_U_4(bp);

	direction = (shim >> 29) & 0x7;
	pkt_type = (shim >> 25) & 0xf;
	pkt_subtype = (shim >> 22) & 0x7;
	li_id = shim & 0x3fffff;

	length -= BCM_LI_SHIM_LEN;
	bp += BCM_LI_SHIM_LEN;

	ND_PRINT("%sBCM-LI-SHIM: direction %s, pkt-type %s, pkt-subtype %s, li-id %u%s",
		 ndo->ndo_vflag ? "\n    " : "",
		 tok2str(bcm_li_direction_values, "unknown", direction),
		 tok2str(bcm_li_pkt_type_values, "unknown", pkt_type),
		 tok2str(bcm_li_pkt_subtype_values, "unknown", pkt_subtype),
		 li_id,
		 ndo->ndo_vflag ? "\n    ": "");

	if (!ndo->ndo_vflag) {
	    ND_TCHECK_LEN(bp, length);
	    return;
	}

	switch (pkt_type) {
	case BCM_LI_PKT_TYPE_ETHERNET:
	    ether_print(ndo, bp, length, ND_BYTES_AVAILABLE_AFTER(bp), NULL, NULL);
	    break;
	case BCM_LI_PKT_TYPE_IPV4:
	    ip_print(ndo, bp, length);
	    break;
	case BCM_LI_PKT_TYPE_IPV6:
	    ip6_print(ndo, bp, length);
	    break;
	case BCM_LI_PKT_TYPE_UNDECIDED:

	    /*
	     * Guess IP version from first nibble.
	     */
	    if ((GET_U_1(bp) >> 4) == 4) {
		ip_print(ndo, bp, length);
	    } else if ((GET_U_1(bp) >> 4) == 6) {
		ip6_print(ndo, bp, length);
	    } else {
		ND_PRINT("unknown payload");
	    }
	    break;

	default:
	    goto invalid;
	}

	return;
invalid:
	nd_print_invalid(ndo);
}

