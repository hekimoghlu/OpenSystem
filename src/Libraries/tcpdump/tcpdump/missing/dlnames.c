/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 13, 2025.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <netdissect-stdinc.h>

#include <pcap.h>
#include <string.h>

#include "pcap-missing.h"
#include "ascii_strcasecmp.h"

struct dlt_choice {
	const char *name;
	const char *description;
	int	dlt;
};

#define DLT_CHOICE(code, description) { #code, description, code }
#define DLT_CHOICE_SENTINEL { NULL, NULL, 0 }

static struct dlt_choice dlt_choices[] = {
	DLT_CHOICE(DLT_NULL, "BSD loopback"),
	DLT_CHOICE(DLT_EN10MB, "Ethernet"),
	DLT_CHOICE(DLT_IEEE802, "Token ring"),
	DLT_CHOICE(DLT_ARCNET, "ARCNET"),
	DLT_CHOICE(DLT_SLIP, "SLIP"),
	DLT_CHOICE(DLT_PPP, "PPP"),
	DLT_CHOICE(DLT_FDDI, "FDDI"),
	DLT_CHOICE(DLT_ATM_RFC1483, "RFC 1483 IP-over-ATM"),
	DLT_CHOICE(DLT_RAW, "Raw IP"),
#ifdef DLT_SLIP_BSDOS
	DLT_CHOICE(DLT_SLIP_BSDOS, "BSD/OS SLIP"),
#endif
#ifdef DLT_PPP_BSDOS
	DLT_CHOICE(DLT_PPP_BSDOS, "BSD/OS PPP"),
#endif
#ifdef DLT_ATM_CLIP
	DLT_CHOICE(DLT_ATM_CLIP, "Linux Classical IP-over-ATM"),
#endif
#ifdef DLT_PPP_SERIAL
	DLT_CHOICE(DLT_PPP_SERIAL, "PPP over serial"),
#endif
#ifdef DLT_PPP_ETHER
	DLT_CHOICE(DLT_PPP_ETHER, "PPPoE"),
#endif
#ifdef DLT_C_HDLC
	DLT_CHOICE(DLT_C_HDLC, "Cisco HDLC"),
#endif
#ifdef DLT_IEEE802_11
	DLT_CHOICE(DLT_IEEE802_11, "802.11"),
#endif
#ifdef DLT_FRELAY
	DLT_CHOICE(DLT_FRELAY, "Frame Relay"),
#endif
#ifdef DLT_LOOP
	DLT_CHOICE(DLT_LOOP, "OpenBSD loopback"),
#endif
#ifdef DLT_ENC
	DLT_CHOICE(DLT_ENC, "OpenBSD encapsulated IP"),
#endif
#ifdef DLT_LINUX_SLL
	DLT_CHOICE(DLT_LINUX_SLL, "Linux cooked"),
#endif
#ifdef DLT_LTALK
	DLT_CHOICE(DLT_LTALK, "Localtalk"),
#endif
#ifdef DLT_PFLOG
	DLT_CHOICE(DLT_PFLOG, "OpenBSD pflog file"),
#endif
#ifdef DLT_PRISM_HEADER
	DLT_CHOICE(DLT_PRISM_HEADER, "802.11 plus Prism header"),
#endif
#ifdef DLT_IP_OVER_FC
	DLT_CHOICE(DLT_IP_OVER_FC, "RFC 2625 IP-over-Fibre Channel"),
#endif
#ifdef DLT_SUNATM
	DLT_CHOICE(DLT_SUNATM, "Sun raw ATM"),
#endif
#ifdef DLT_IEEE802_11_RADIO
	DLT_CHOICE(DLT_IEEE802_11_RADIO, "802.11 plus radio information header"),
#endif
#ifdef DLT_ARCNET_LINUX
	DLT_CHOICE(DLT_ARCNET_LINUX, "Linux ARCNET"),
#endif
#ifdef DLT_LINUX_IRDA
	DLT_CHOICE(DLT_LINUX_IRDA, "Linux IrDA"),
#endif
#ifdef DLT_CIP
	DLT_CHOICE(DLT_CIP, "Linux Classical IP-over-ATM"),
#endif
#ifdef DLT_HDLC
	DLT_CHOICE(DLT_HDLC, "Cisco HDLC"),
#endif
	DLT_CHOICE_SENTINEL
};

#ifndef HAVE_PCAP_DATALINK_NAME_TO_VAL
int
pcap_datalink_name_to_val(const char *name)
{
	int i;

	for (i = 0; dlt_choices[i].name != NULL; i++) {
		if (ascii_strcasecmp(dlt_choices[i].name + sizeof("DLT_") - 1,
		    name) == 0)
			return (dlt_choices[i].dlt);
	}
	return (-1);
}

const char *
pcap_datalink_val_to_name(int dlt)
{
	int i;

	for (i = 0; dlt_choices[i].name != NULL; i++) {
		if (dlt_choices[i].dlt == dlt)
			return (dlt_choices[i].name + sizeof("DLT_") - 1);
	}
	return (NULL);
}
#endif

const char *
pcap_datalink_val_to_description(int dlt)
{
	int i;

	for (i = 0; dlt_choices[i].name != NULL; i++) {
		if (dlt_choices[i].dlt == dlt)
			return (dlt_choices[i].description);
	}
	return (NULL);
}
