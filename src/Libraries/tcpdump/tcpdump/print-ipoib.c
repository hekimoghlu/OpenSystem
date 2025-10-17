/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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
/*
 * Copyright (c) 2011, 2016, Oracle and/or its affiliates. All rights reserved.
 */

/* \summary: IP-over-InfiniBand (IPoIB) printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"
#include "extract.h"
#include "addrtoname.h"


extern const struct tok ethertype_values[];

#define	IPOIB_HDRLEN	44

static inline void
ipoib_hdr_print(netdissect_options *ndo, const u_char *bp, u_int length)
{
	uint16_t ether_type;

	ether_type = GET_BE_U_2(bp + 40);
	if (!ndo->ndo_qflag) {
		ND_PRINT(", ethertype %s (0x%04x)",
			     tok2str(ethertype_values,"Unknown", ether_type),
			     ether_type);
	} else {
		ND_PRINT(", ethertype %s",
			     tok2str(ethertype_values,"Unknown", ether_type));
	}

	ND_PRINT(", length %u: ", length);
}

/*
 * Print an InfiniBand frame.
 * This might be encapsulated within another frame; we might be passed
 * a pointer to a function that can print header information for that
 * frame's protocol, and an argument to pass to that function.
 */
static void
ipoib_print(netdissect_options *ndo, const u_char *p, u_int length, u_int caplen,
    void (*print_encap_header)(const u_char *), const u_char *encap_header_arg)
{
	const u_char *orig_hdr = p;
	u_int orig_length;
	u_short ether_type;

	if (caplen < IPOIB_HDRLEN) {
		nd_print_trunc(ndo);
		ndo->ndo_ll_hdr_len += caplen;
		return;
	}

	if (length < IPOIB_HDRLEN) {
		nd_print_trunc(ndo);
		ndo->ndo_ll_hdr_len += length;
		return;
	}

	if (ndo->ndo_eflag) {
		nd_print_protocol_caps(ndo);
		if (print_encap_header != NULL)
			(*print_encap_header)(encap_header_arg);
		ipoib_hdr_print(ndo, p, length);
	}
	orig_length = length;

	ndo->ndo_ll_hdr_len += IPOIB_HDRLEN;
	length -= IPOIB_HDRLEN;
	caplen -= IPOIB_HDRLEN;
	ether_type = GET_BE_U_2(p + 40);
	p += IPOIB_HDRLEN;

	if (ethertype_print(ndo, ether_type, p, length, caplen, NULL, NULL) == 0) {
		/* ether_type not known, print raw packet */
		if (!ndo->ndo_eflag) {
			if (print_encap_header != NULL)
				(*print_encap_header)(encap_header_arg);
			ipoib_hdr_print(ndo, orig_hdr , orig_length);
		}

		if (!ndo->ndo_suppress_default_print)
			ND_DEFAULTPRINT(p, caplen);
	}
}

/*
 * This is the top level routine of the printer.  'p' points
 * to the ether header of the packet, 'h->ts' is the timestamp,
 * 'h->len' is the length of the packet off the wire, and 'h->caplen'
 * is the number of bytes actually captured.
 */
void
ipoib_if_print(netdissect_options *ndo, const struct pcap_pkthdr *h, const u_char *p)
{
	ndo->ndo_protocol = "ipoib";
	ipoib_print(ndo, p, h->len, h->caplen, NULL, NULL);
}
