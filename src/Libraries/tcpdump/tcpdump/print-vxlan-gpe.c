/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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
/* \summary: Generic Protocol Extension for VXLAN (VXLAN GPE) printer */

/* specification: draft-ietf-nvo3-vxlan-gpe-10 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "extract.h"

static const struct tok vxlan_gpe_flags [] = {
    { 0x08, "I" },
    { 0x04, "P" },
    { 0x02, "B" },
    { 0x01, "O" },
    { 0, NULL }
};

#define VXLAN_GPE_HDR_LEN 8

/*
 * VXLAN GPE header, draft-ietf-nvo3-vxlan-gpe-01
 *                   Generic Protocol Extension for VXLAN
 *
 *     0                   1                   2                   3
 *     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |R|R|Ver|I|P|R|O|       Reserved                |Next Protocol  |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *    |                VXLAN Network Identifier (VNI) |   Reserved    |
 *    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */

void
vxlan_gpe_print(netdissect_options *ndo, const u_char *bp, u_int len)
{
    uint8_t flags;
    uint8_t next_protocol;
    uint32_t vni;

    ndo->ndo_protocol = "vxlan_gpe";
    ND_PRINT("VXLAN-GPE, ");
    if (len < VXLAN_GPE_HDR_LEN) {
        ND_PRINT(" (len %u < %u)", len, VXLAN_GPE_HDR_LEN);
        goto invalid;
    }

    flags = GET_U_1(bp);
    bp += 1;
    len -= 1;
    ND_PRINT("flags [%s], ",
              bittok2str_nosep(vxlan_gpe_flags, "none", flags));

    /* Reserved */
    bp += 2;
    len -= 2;

    next_protocol = GET_U_1(bp);
    bp += 1;
    len -= 1;

    vni = GET_BE_U_3(bp);
    bp += 3;
    len -= 3;

    /* Reserved */
    ND_TCHECK_1(bp);
    bp += 1;
    len -= 1;

    ND_PRINT("vni %u", vni);
    ND_PRINT(ndo->ndo_vflag ? "\n    " : ": ");

    switch (next_protocol) {
    case 0x1:
        ip_print(ndo, bp, len);
        break;
    case 0x2:
        ip6_print(ndo, bp, len);
        break;
    case 0x3:
        ether_print(ndo, bp, len, ND_BYTES_AVAILABLE_AFTER(bp), NULL, NULL);
        break;
    case 0x4:
        nsh_print(ndo, bp, len);
        break;
    default:
        ND_PRINT("ERROR: unknown-next-protocol");
        goto invalid;
    }

	return;

invalid:
    nd_print_invalid(ndo);
}

