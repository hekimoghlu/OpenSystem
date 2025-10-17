/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
/* \summary: PPP Van Jacobson compression printer */

/* specification: RFC 1144 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"
#include "extract.h"
#include "slcompress.h"
#include "ppp.h"

/*
 * XXX - for BSD/OS PPP, what packets get supplied with a PPP header type
 * of PPP_VJC and what packets get supplied with a PPP header type of
 * PPP_VJNC?  PPP_VJNC is for "UNCOMPRESSED_TCP" packets, and PPP_VJC
 * is for COMPRESSED_TCP packets (PPP_IP is used for TYPE_IP packets).
 *
 * RFC 1144 implies that, on the wire, the packet type is *not* needed
 * for PPP, as different PPP protocol types can be used; it only needs
 * to be put on the wire for SLIP.
 *
 * It also indicates that, for compressed SLIP:
 *
 *	If the COMPRESSED_TCP bit is set in the first byte, it's
 *	a COMPRESSED_TCP packet; that byte is the change byte, and
 *	the COMPRESSED_TCP bit, 0x80, isn't used in the change byte.
 *
 *	If the upper 4 bits of the first byte are 7, it's an
 *	UNCOMPRESSED_TCP packet; that byte is the first byte of
 *	the UNCOMPRESSED_TCP modified IP header, with a connection
 *	number in the protocol field, and with the version field
 *	being 7, not 4.
 *
 *	Otherwise, the packet is an IPv4 packet (where the upper 4 bits
 *	of the packet are 4).
 *
 * So this routine looks as if it's sort-of intended to handle
 * compressed SLIP, although it doesn't handle UNCOMPRESSED_TCP
 * correctly for that (it doesn't fix the version number and doesn't
 * do anything to the protocol field), and doesn't check for COMPRESSED_TCP
 * packets correctly for that (you only check the first bit - see
 * B.1 in RFC 1144).
 *
 * But it's called for BSD/OS PPP, not SLIP - perhaps BSD/OS does weird
 * things with the headers?
 *
 * Without a BSD/OS VJC-compressed PPP trace, or knowledge of what the
 * BSD/OS VJC code does, we can't say what's the case.
 *
 * We therefore leave "proto" - which is the PPP protocol type - in place,
 * *not* marked as unused, for now, so that GCC warnings about the
 * unused argument remind us that we should fix this some day.
 *
 * XXX - also, it fetches the TCP checksum field in COMPRESSED_TCP
 * packets with GET_HE_U_2, rather than with GET_BE_U_2(); RFC 1144 says
 * it's "the unmodified TCP checksum", which would imply that it's
 * big-endian, but perhaps, on the platform where this was developed,
 * the packets were munged by the networking stack before being handed
 * to the packet capture mechanism.
 */
int
vjc_print(netdissect_options *ndo, const u_char *bp, u_short proto _U_)
{
	int i;

	ndo->ndo_protocol = "vjc";
	switch (GET_U_1(bp) & 0xf0) {
	case TYPE_IP:
		if (ndo->ndo_eflag)
			ND_PRINT("(vjc type=IP) ");
		return PPP_IP;
	case TYPE_UNCOMPRESSED_TCP:
		if (ndo->ndo_eflag)
			ND_PRINT("(vjc type=raw TCP) ");
		return PPP_IP;
	case TYPE_COMPRESSED_TCP:
		if (ndo->ndo_eflag)
			ND_PRINT("(vjc type=compressed TCP) ");
		for (i = 0; i < 8; i++) {
			if (GET_U_1(bp + 1) & (0x80 >> i))
				ND_PRINT("%c", "?CI?SAWU"[i]);
		}
		if (GET_U_1(bp + 1))
			ND_PRINT(" ");
		ND_PRINT("C=0x%02x ", GET_U_1(bp + 2));
		ND_PRINT("sum=0x%04x ", GET_HE_U_2(bp + 3));
		return -1;
	case TYPE_ERROR:
		if (ndo->ndo_eflag)
			ND_PRINT("(vjc type=error) ");
		return -1;
	default:
		if (ndo->ndo_eflag)
			ND_PRINT("(vjc type=0x%02x) ", GET_U_1(bp) & 0xf0);
		return -1;
	}
}
