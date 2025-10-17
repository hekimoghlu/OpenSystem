/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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
/* \summary: unsupported link-layer protocols printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"

void
unsupported_if_print(netdissect_options *ndo, const struct pcap_pkthdr *h,
		     const u_char *p)
{
	ndo->ndo_protocol = "unsupported";
	nd_print_protocol_caps(ndo);
#ifdef __APPLE__
	hex_and_ascii_print(ndo, "\n\t", p, MIN(h->caplen, ndo->ndo_snaplen));
#else /* __APPLE__ */
	hex_and_ascii_print(ndo, "\n\t", p, h->caplen);
#endif /* __APPLE__ */
}
