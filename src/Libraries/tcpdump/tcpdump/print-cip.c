/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
/* \summary: Linux Classical IP over ATM printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "addrtoname.h"

static const unsigned char rfcllc[] = {
	0xaa,	/* DSAP: non-ISO */
	0xaa,	/* SSAP: non-ISO */
	0x03,	/* Ctrl: Unnumbered Information Command PDU */
	0x00,	/* OUI: EtherType */
	0x00,
	0x00 };

/*
 * This is the top level routine of the printer.  'p' points
 * to the LLC/SNAP or raw header of the packet, 'h->ts' is the timestamp,
 * 'h->len' is the length of the packet off the wire, and 'h->caplen'
 * is the number of bytes actually captured.
 */
void
cip_if_print(netdissect_options *ndo, const struct pcap_pkthdr *h, const u_char *p)
{
	u_int caplen = h->caplen;
	u_int length = h->len;
	int llc_hdrlen;

	ndo->ndo_protocol = "cip";

	if (ndo->ndo_eflag)
		/*
		 * There is no MAC-layer header, so just print the length.
		 */
		ND_PRINT("%u: ", length);

	ND_TCHECK_LEN(p, sizeof(rfcllc));
	if (memcmp(rfcllc, p, sizeof(rfcllc)) == 0) {
		/*
		 * LLC header is present.  Try to print it & higher layers.
		 */
		llc_hdrlen = llc_print(ndo, p, length, caplen, NULL, NULL);
		if (llc_hdrlen < 0) {
			/* packet type not known, print raw packet */
			if (!ndo->ndo_suppress_default_print)
				ND_DEFAULTPRINT(p, caplen);
			llc_hdrlen = -llc_hdrlen;
		}
	} else {
		/*
		 * LLC header is absent; treat it as just IP.
		 */
		llc_hdrlen = 0;
		ip_print(ndo, p, length);
	}

	ndo->ndo_ll_hdr_len += llc_hdrlen;
}
