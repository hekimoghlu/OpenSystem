/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
/* \summary: Secure Shell (SSH) printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"
#include "netdissect-ctype.h"

#include "netdissect.h"
#include "extract.h"

static int
ssh_print_version(netdissect_options *ndo, const u_char *pptr, u_int len)
{
	u_int idx = 0;

	if ( GET_U_1(pptr+idx) != 'S' )
		return 0;
	idx++;
	if ( GET_U_1(pptr+idx) != 'S' )
		return 0;
	idx++;
	if ( GET_U_1(pptr+idx) != 'H' )
		return 0;
	idx++;
	if ( GET_U_1(pptr+idx) != '-' )
		return 0;
	idx++;

	while (idx < len) {
		u_char c;

		c = GET_U_1(pptr + idx);
		if (c == '\n') {
			/*
			 * LF without CR; end of line.
			 * Skip the LF and print the line, with the
			 * exception of the LF.
			 */
			goto print;
		} else if (c == '\r') {
			/* CR - any LF? */
			if ((idx+1) >= len) {
				/* not in this packet */
				goto trunc;
			}
			if (GET_U_1(pptr + idx + 1) == '\n') {
				/*
				 * CR-LF; end of line.
				 * Skip the CR-LF and print the line, with
				 * the exception of the CR-LF.
				 */
				goto print;
			}

			/*
			 * CR followed by something else; treat this as
			 * if it were binary data and don't print it.
			 */
			goto trunc;
		} else if (!ND_ASCII_ISPRINT(c) ) {
			/*
			 * Not a printable ASCII character; treat this
			 * as if it were binary data and don't print it.
			 */
			goto trunc;
		}
		idx++;
	}
trunc:
	return -1;
print:
	ND_PRINT(": ");
	nd_print_protocol_caps(ndo);
	ND_PRINT(": %.*s", (int)idx, pptr);
	return idx;
}

void
ssh_print(netdissect_options *ndo, const u_char *pptr, u_int len)
{
	ndo->ndo_protocol = "ssh";

	ssh_print_version(ndo, pptr, len);
}
