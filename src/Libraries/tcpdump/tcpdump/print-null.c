/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
/* \summary: BSD loopback device printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "extract.h"
#include "af.h"


/*
 * The DLT_NULL packet header is 4 bytes long. It contains a host-byte-order
 * 32-bit integer that specifies the family, e.g. AF_INET.
 *
 * Note here that "host" refers to the host on which the packets were
 * captured; that isn't necessarily *this* host.
 *
 * The OpenBSD DLT_LOOP packet header is the same, except that the integer
 * is in network byte order.
 */
#define	NULL_HDRLEN 4

/*
 * Byte-swap a 32-bit number.
 * ("htonl()" or "ntohl()" won't work - we want to byte-swap even on
 * big-endian platforms.)
 */
#define	SWAPLONG(y) \
((((y)&0xff)<<24) | (((y)&0xff00)<<8) | (((y)&0xff0000)>>8) | (((y)>>24)&0xff))

static void
null_hdr_print(netdissect_options *ndo, uint32_t family, u_int length)
{
	if (!ndo->ndo_qflag) {
		ND_PRINT("AF %s (%u)",
			tok2str(bsd_af_values,"Unknown",family),family);
	} else {
		ND_PRINT("%s",
			tok2str(bsd_af_values,"Unknown AF %u",family));
	}

	ND_PRINT(", length %u: ", length);
}

/*
 * This is the top level routine of the printer.  'p' points
 * to the ether header of the packet, 'h->ts' is the timestamp,
 * 'h->len' is the length of the packet off the wire, and 'h->caplen'
 * is the number of bytes actually captured.
 */
void
null_if_print(netdissect_options *ndo, const struct pcap_pkthdr *h, const u_char *p)
{
	u_int length = h->len;
	u_int caplen = h->caplen;
	uint32_t family;

	ndo->ndo_protocol = "null";
	ND_TCHECK_LEN(p, NULL_HDRLEN);
	ndo->ndo_ll_hdr_len += NULL_HDRLEN;

	family = GET_HE_U_4(p);

	/*
	 * This isn't necessarily in our host byte order; if this is
	 * a DLT_LOOP capture, it's in network byte order, and if
	 * this is a DLT_NULL capture from a machine with the opposite
	 * byte-order, it's in the opposite byte order from ours.
	 *
	 * If the upper 16 bits aren't all zero, assume it's byte-swapped.
	 */
	if ((family & 0xFFFF0000) != 0)
		family = SWAPLONG(family);

	if (ndo->ndo_eflag)
		null_hdr_print(ndo, family, length);

	length -= NULL_HDRLEN;
	caplen -= NULL_HDRLEN;
	p += NULL_HDRLEN;

	switch (family) {

	case BSD_AFNUM_INET:
		ip_print(ndo, p, length);
		break;

	case BSD_AFNUM_INET6_BSD:
	case BSD_AFNUM_INET6_FREEBSD:
	case BSD_AFNUM_INET6_DARWIN:
		ip6_print(ndo, p, length);
		break;

	case BSD_AFNUM_ISO:
		isoclns_print(ndo, p, length);
		break;

	case BSD_AFNUM_APPLETALK:
		atalk_print(ndo, p, length);
		break;

	case BSD_AFNUM_IPX:
		ipx_print(ndo, p, length);
		break;

	default:
		/* unknown AF_ value */
		if (!ndo->ndo_eflag)
			null_hdr_print(ndo, family, length + NULL_HDRLEN);
		if (!ndo->ndo_suppress_default_print)
			ND_DEFAULTPRINT(p, caplen);
	}
}
