/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 10, 2022.
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
/* \summary: Communication access for land mobiles (CALM) printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "extract.h"
#include "addrtoname.h"

/*
   ISO 29281:2009
   Intelligent Transport Systems . Communications access for land mobiles (CALM)
   CALM non-IP networking
*/

/*
 * This is the top level routine of the printer.  'bp' points
 * to the calm header of the packet.
 */
void
calm_fast_print(netdissect_options *ndo, const u_char *bp, u_int length, const struct lladdr_info *src)
{
	ndo->ndo_protocol = "calm_fast";

	ND_PRINT("CALM FAST");
	if (src != NULL)
		ND_PRINT(" src:%s", (src->addr_string)(ndo, src->addr));
	ND_PRINT("; ");

	if (length < 2) {
		ND_PRINT(" (length %u < 2)", length);
		goto invalid;
	}

	ND_PRINT("SrcNwref:%u; ", GET_U_1(bp));
	length -= 1;
	bp += 1;

	ND_PRINT("DstNwref:%u; ", GET_U_1(bp));
	length -= 1;
	bp += 1;

	if (ndo->ndo_vflag)
		ND_DEFAULTPRINT(bp, length);
	return;

invalid:
	nd_print_invalid(ndo);
	ND_TCHECK_LEN(bp, length);
}
