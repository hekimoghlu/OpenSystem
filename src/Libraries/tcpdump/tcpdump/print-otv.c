/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
/* \summary: Overlay Transport Virtualization (OTV) printer */

/* specification: draft-hasmit-otv-04 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "extract.h"

#define OTV_HDR_LEN 8

/*
 * OTV header, draft-hasmit-otv-04
 *
 *     0                   1                   2                   3
 *     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 *     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *     |R|R|R|R|I|R|R|R|           Overlay ID                          |
 *     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *     |          Instance ID                          | Reserved      |
 *     +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */

void
otv_print(netdissect_options *ndo, const u_char *bp, u_int len)
{
    uint8_t flags;

    ndo->ndo_protocol = "otv";
    ND_PRINT("OTV, ");
    if (len < OTV_HDR_LEN) {
        ND_PRINT("[length %u < %u]", len, OTV_HDR_LEN);
        goto invalid;
    }

    flags = GET_U_1(bp);
    ND_PRINT("flags [%s] (0x%02x), ", flags & 0x08 ? "I" : ".", flags);
    bp += 1;

    ND_PRINT("overlay %u, ", GET_BE_U_3(bp));
    bp += 3;

    ND_PRINT("instance %u\n", GET_BE_U_3(bp));
    bp += 3;

    /* Reserved */
    ND_TCHECK_1(bp);
    bp += 1;

    ether_print(ndo, bp, len - OTV_HDR_LEN, ND_BYTES_AVAILABLE_AFTER(bp), NULL, NULL);
    return;

invalid:
    nd_print_invalid(ndo);
    ND_TCHECK_LEN(bp, len);
}
