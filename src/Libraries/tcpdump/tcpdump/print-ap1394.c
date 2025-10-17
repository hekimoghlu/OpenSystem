/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
/* \summary: Apple IP-over-IEEE 1394 printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "extract.h"
#include "addrtoname.h"
#include "ethertype.h"

/*
 * Structure of a header for Apple's IP-over-IEEE 1384 BPF header.
 */
#define FIREWIRE_EUI64_LEN	8
struct firewire_header {
	nd_byte     firewire_dhost[FIREWIRE_EUI64_LEN];
	nd_byte     firewire_shost[FIREWIRE_EUI64_LEN];
	nd_uint16_t firewire_type;
};

/*
 * Length of that header; note that some compilers may pad
 * "struct firewire_header" to a multiple of 4 bytes, for example, so
 * "sizeof (struct firewire_header)" may not give the right answer.
 */
#define FIREWIRE_HDRLEN		18

static const char *
fwaddr_string(netdissect_options *ndo, const u_char *addr)
{
	return GET_LINKADDR_STRING(addr, LINKADDR_IEEE1394, FIREWIRE_EUI64_LEN);
}

static void
ap1394_hdr_print(netdissect_options *ndo, const u_char *bp, u_int length)
{
	const struct firewire_header *fp;
	uint16_t firewire_type;

	fp = (const struct firewire_header *)bp;

	ND_PRINT("%s > %s",
		     fwaddr_string(ndo, fp->firewire_shost),
		     fwaddr_string(ndo, fp->firewire_dhost));

	firewire_type = GET_BE_U_2(fp->firewire_type);
	if (!ndo->ndo_qflag) {
		ND_PRINT(", ethertype %s (0x%04x)",
			       tok2str(ethertype_values,"Unknown", firewire_type),
                               firewire_type);
        } else {
                ND_PRINT(", %s", tok2str(ethertype_values,"Unknown Ethertype (0x%04x)", firewire_type));
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
ap1394_if_print(netdissect_options *ndo, const struct pcap_pkthdr *h, const u_char *p)
{
	u_int length = h->len;
	u_int caplen = h->caplen;
	const struct firewire_header *fp;
	u_short ether_type;
	struct lladdr_info src, dst;

	ndo->ndo_protocol = "ap1394";
	ND_TCHECK_LEN(p, FIREWIRE_HDRLEN);
	ndo->ndo_ll_hdr_len += FIREWIRE_HDRLEN;

	if (ndo->ndo_eflag)
		ap1394_hdr_print(ndo, p, length);

	length -= FIREWIRE_HDRLEN;
	caplen -= FIREWIRE_HDRLEN;
	fp = (const struct firewire_header *)p;
	p += FIREWIRE_HDRLEN;

	ether_type = GET_BE_U_2(fp->firewire_type);
	src.addr = fp->firewire_shost;
	src.addr_string = fwaddr_string;
	dst.addr = fp->firewire_dhost;
	dst.addr_string = fwaddr_string;
	if (ethertype_print(ndo, ether_type, p, length, caplen, &src, &dst) == 0) {
		/* ether_type not known, print raw packet */
		if (!ndo->ndo_eflag)
			ap1394_hdr_print(ndo, (const u_char *)fp, length + FIREWIRE_HDRLEN);

		if (!ndo->ndo_suppress_default_print)
			ND_DEFAULTPRINT(p, caplen);
	}
}
