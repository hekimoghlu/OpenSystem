/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
/* \summary: Bluetooth printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#define ND_LONGJMP_FROM_TCHECK
#include "netdissect.h"
#include "extract.h"

#ifdef DLT_BLUETOOTH_HCI_H4_WITH_PHDR

/*
 * Header prepended by libpcap to each bluetooth h4 frame;
 * the direction field is in network byte order.
 */
typedef struct _bluetooth_h4_header {
	nd_uint32_t direction; /* if first bit is set direction is incoming */
} bluetooth_h4_header;

#define	BT_HDRLEN sizeof(bluetooth_h4_header)

/*
 * This is the top level routine of the printer.  'p' points
 * to the bluetooth header of the packet, 'h->ts' is the timestamp,
 * 'h->len' is the length of the packet off the wire, and 'h->caplen'
 * is the number of bytes actually captured.
 */
void
bt_if_print(netdissect_options *ndo, const struct pcap_pkthdr *h, const u_char *p)
{
	u_int length = h->len;
	u_int caplen = h->caplen;
	const bluetooth_h4_header* hdr = (const bluetooth_h4_header*)p;

	ndo->ndo_protocol = "bluetooth";
	nd_print_protocol(ndo);
	ND_TCHECK_LEN(p, BT_HDRLEN);
	ndo->ndo_ll_hdr_len += BT_HDRLEN;
	caplen -= BT_HDRLEN;
	length -= BT_HDRLEN;
	p += BT_HDRLEN;
	if (ndo->ndo_eflag)
		ND_PRINT(", hci length %u, direction %s", length,
			 (GET_BE_U_4(hdr->direction)&0x1) ? "in" : "out");

	if (!ndo->ndo_suppress_default_print)
		ND_DEFAULTPRINT(p, caplen);
}
#endif
