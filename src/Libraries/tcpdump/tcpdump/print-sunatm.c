/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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
/* \summary: SunATM DLPI capture printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "extract.h"

#include "atm.h"

/* SunATM header for ATM packet */
#define DIR_POS		0	/* Direction (0x80 = transmit, 0x00 = receive) */
#define VPI_POS		1	/* VPI */
#define VCI_POS		2	/* VCI */
#define PKT_BEGIN_POS   4	/* Start of the ATM packet */

/* Protocol type values in the bottom for bits of the byte at SUNATM_DIR_POS. */
#define PT_LANE		0x01	/* LANE */
#define PT_LLC		0x02	/* LLC encapsulation */

/*
 * This is the top level routine of the printer.  'p' points
 * to the SunATM pseudo-header for the packet, 'h->ts' is the timestamp,
 * 'h->len' is the length of the packet off the wire, and 'h->caplen'
 * is the number of bytes actually captured.
 */
void
sunatm_if_print(netdissect_options *ndo,
                const struct pcap_pkthdr *h, const u_char *p)
{
	u_int caplen = h->caplen;
	u_int length = h->len;
	u_short vci;
	u_char vpi;
	u_int traftype;

	ndo->ndo_protocol = "sunatm";

	if (ndo->ndo_eflag) {
		ND_PRINT(GET_U_1(p + DIR_POS) & 0x80 ? "Tx: " : "Rx: ");
	}

	switch (GET_U_1(p + DIR_POS) & 0x0f) {

	case PT_LANE:
		traftype = ATM_LANE;
		break;

	case PT_LLC:
		traftype = ATM_LLC;
		break;

	default:
		traftype = ATM_UNKNOWN;
		break;
	}

	vpi = GET_U_1(p + VPI_POS);
	vci = GET_BE_U_2(p + VCI_POS);

	p += PKT_BEGIN_POS;
	caplen -= PKT_BEGIN_POS;
	length -= PKT_BEGIN_POS;
	ndo->ndo_ll_hdr_len += PKT_BEGIN_POS;
	atm_print(ndo, vpi, vci, traftype, p, length, caplen);
}
