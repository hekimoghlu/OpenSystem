/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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
/* \summary: Symantec Enterprise Firewall printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "extract.h"
#include "ethertype.h"

struct symantec_header {
	nd_byte     stuff1[6];
	nd_uint16_t ether_type;
	nd_byte     stuff2[36];
};

static void
symantec_hdr_print(netdissect_options *ndo, const u_char *bp, u_int length)
{
	const struct symantec_header *sp;
	uint16_t etype;

	sp = (const struct symantec_header *)bp;

	etype = GET_BE_U_2(sp->ether_type);
	if (!ndo->ndo_qflag) {
	        if (etype <= MAX_ETHERNET_LENGTH_VAL)
		          ND_PRINT("invalid ethertype %u", etype);
                else
		          ND_PRINT("ethertype %s (0x%04x)",
				       tok2str(ethertype_values,"Unknown", etype),
                                       etype);
        } else {
                if (etype <= MAX_ETHERNET_LENGTH_VAL)
                          ND_PRINT("invalid ethertype %u", etype);
                else
                          ND_PRINT("%s", tok2str(ethertype_values,"Unknown Ethertype (0x%04x)", etype));
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
symantec_if_print(netdissect_options *ndo, const struct pcap_pkthdr *h, const u_char *p)
{
	u_int length = h->len;
	u_int caplen = h->caplen;
	const struct symantec_header *sp;
	u_short ether_type;

	ndo->ndo_protocol = "symantec";
	ND_TCHECK_LEN(p, sizeof(struct symantec_header));

	ndo->ndo_ll_hdr_len += sizeof (struct symantec_header);
	if (ndo->ndo_eflag)
		symantec_hdr_print(ndo, p, length);

	length -= sizeof (struct symantec_header);
	caplen -= sizeof (struct symantec_header);
	sp = (const struct symantec_header *)p;
	p += sizeof (struct symantec_header);

	ether_type = GET_BE_U_2(sp->ether_type);

	if (ether_type <= MAX_ETHERNET_LENGTH_VAL) {
		/* ether_type not known, print raw packet */
		if (!ndo->ndo_eflag)
			symantec_hdr_print(ndo, (const u_char *)sp, length + sizeof (struct symantec_header));

		if (!ndo->ndo_suppress_default_print)
			ND_DEFAULTPRINT(p, caplen);
	} else if (ethertype_print(ndo, ether_type, p, length, caplen, NULL, NULL) == 0) {
		/* ether_type not known, print raw packet */
		if (!ndo->ndo_eflag)
			symantec_hdr_print(ndo, (const u_char *)sp, length + sizeof (struct symantec_header));

		if (!ndo->ndo_suppress_default_print)
			ND_DEFAULTPRINT(p, caplen);
	}
}
