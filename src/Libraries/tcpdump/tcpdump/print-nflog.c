/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 30, 2021.
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
/* \summary: DLT_NFLOG printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"
#include "extract.h"

#ifdef DLT_NFLOG

/*
 * Structure of an NFLOG header and TLV parts, as described at
 * https://www.tcpdump.org/linktypes/LINKTYPE_NFLOG.html
 *
 * The NFLOG header is big-endian.
 *
 * The TLV length and type are in host byte order.  The value is either
 * big-endian or is an array of bytes in some externally-specified byte
 * order (text string, link-layer address, link-layer header, packet
 * data, etc.).
 */
typedef struct nflog_hdr {
	nd_uint8_t	nflog_family;		/* address family */
	nd_uint8_t	nflog_version;		/* version */
	nd_uint16_t	nflog_rid;		/* resource ID */
} nflog_hdr_t;

#define NFLOG_HDR_LEN sizeof(nflog_hdr_t)

typedef struct nflog_tlv {
	nd_uint16_t	tlv_length;		/* tlv length */
	nd_uint16_t	tlv_type;		/* tlv type */
	/* value follows this */
} nflog_tlv_t;

#define NFLOG_TLV_LEN sizeof(nflog_tlv_t)

typedef struct nflog_packet_hdr {
	nd_uint16_t	hw_protocol;	/* hw protocol */
	nd_uint8_t	hook;		/* netfilter hook */
	nd_byte		pad[1];		/* padding to 32 bits */
} nflog_packet_hdr_t;

typedef struct nflog_hwaddr {
	nd_uint16_t	hw_addrlen;	/* address length */
	nd_byte		pad[2];		/* padding to 32-bit boundary */
	nd_byte		hw_addr[8];	/* address, up to 8 bytes */
} nflog_hwaddr_t;

typedef struct nflog_timestamp {
	nd_uint64_t	sec;
	nd_uint64_t	usec;
} nflog_timestamp_t;

/*
 * TLV types.
 */
#define NFULA_PACKET_HDR		1	/* nflog_packet_hdr_t */
#define NFULA_MARK			2	/* packet mark from skbuff */
#define NFULA_TIMESTAMP			3	/* nflog_timestamp_t for skbuff's time stamp */
#define NFULA_IFINDEX_INDEV		4	/* ifindex of device on which packet received (possibly bridge group) */
#define NFULA_IFINDEX_OUTDEV		5	/* ifindex of device on which packet transmitted (possibly bridge group) */
#define NFULA_IFINDEX_PHYSINDEV		6	/* ifindex of physical device on which packet received (not bridge group) */
#define NFULA_IFINDEX_PHYSOUTDEV	7	/* ifindex of physical device on which packet transmitted (not bridge group) */
#define NFULA_HWADDR			8	/* nflog_hwaddr_t for hardware address */
#define NFULA_PAYLOAD			9	/* packet payload */
#define NFULA_PREFIX			10	/* text string - null-terminated, count includes NUL */
#define NFULA_UID			11	/* UID owning socket on which packet was sent/received */
#define NFULA_SEQ			12	/* sequence number of packets on this NFLOG socket */
#define NFULA_SEQ_GLOBAL		13	/* sequence number of pakets on all NFLOG sockets */
#define NFULA_GID			14	/* GID owning socket on which packet was sent/received */
#define NFULA_HWTYPE			15	/* ARPHRD_ type of skbuff's device */
#define NFULA_HWHEADER			16	/* skbuff's MAC-layer header */
#define NFULA_HWLEN			17	/* length of skbuff's MAC-layer header */

static const struct tok nflog_values[] = {
	{ AF_INET,		"IPv4" },
	{ AF_INET6,		"IPv6" },
	{ 0,			NULL }
};

static void
nflog_hdr_print(netdissect_options *ndo, const nflog_hdr_t *hdr, u_int length)
{
	ND_PRINT("version %u, resource ID %u",
	    GET_U_1(hdr->nflog_version), GET_BE_U_2(hdr->nflog_rid));

	if (!ndo->ndo_qflag) {
		ND_PRINT(", family %s (%u)",
			 tok2str(nflog_values, "Unknown",
				 GET_U_1(hdr->nflog_family)),
			 GET_U_1(hdr->nflog_family));
		} else {
		ND_PRINT(", %s",
			 tok2str(nflog_values,
				 "Unknown NFLOG (0x%02x)",
			 GET_U_1(hdr->nflog_family)));
		}

	ND_PRINT(", length %u: ", length);
}

void
nflog_if_print(netdissect_options *ndo,
	       const struct pcap_pkthdr *h, const u_char *p)
{
	const nflog_hdr_t *hdr = (const nflog_hdr_t *)p;
	uint16_t size;
	uint16_t h_size = NFLOG_HDR_LEN;
	u_int caplen = h->caplen;
	u_int length = h->len;

	ndo->ndo_protocol = "nflog";
	if (caplen < NFLOG_HDR_LEN) {
		nd_print_trunc(ndo);
		ndo->ndo_ll_hdr_len += caplen;
		return;
	}
	ndo->ndo_ll_hdr_len += NFLOG_HDR_LEN;

	ND_TCHECK_SIZE(hdr);
	if (GET_U_1(hdr->nflog_version) != 0) {
		ND_PRINT("version %u (unknown)", GET_U_1(hdr->nflog_version));
		return;
	}

	if (ndo->ndo_eflag)
		nflog_hdr_print(ndo, hdr, length);

	p += NFLOG_HDR_LEN;
	length -= NFLOG_HDR_LEN;
	caplen -= NFLOG_HDR_LEN;

	while (length > 0) {
		const nflog_tlv_t *tlv;

		/* We have some data.  Do we have enough for the TLV header? */
		if (caplen < NFLOG_TLV_LEN)
			goto trunc;	/* No. */

		tlv = (const nflog_tlv_t *) p;
		ND_TCHECK_SIZE(tlv);
		size = GET_HE_U_2(tlv->tlv_length);
		if (size % 4 != 0)
			size += 4 - size % 4;

		/* Is the TLV's length less than the minimum? */
		if (size < NFLOG_TLV_LEN)
			goto trunc;	/* Yes. Give up now. */

		/* Do we have enough data for the full TLV? */
		if (caplen < size)
			goto trunc;	/* No. */

		if (GET_HE_U_2(tlv->tlv_type) == NFULA_PAYLOAD) {
			/*
			 * This TLV's data is the packet payload.
			 * Skip past the TLV header, and break out
			 * of the loop so we print the packet data.
			 */
			p += NFLOG_TLV_LEN;
			h_size += NFLOG_TLV_LEN;
			length -= NFLOG_TLV_LEN;
			caplen -= NFLOG_TLV_LEN;
			break;
		}

		p += size;
		h_size += size;
		length -= size;
		caplen -= size;
	}

	switch (GET_U_1(hdr->nflog_family)) {

	case AF_INET:
		ip_print(ndo, p, length);
		break;

	case AF_INET6:
		ip6_print(ndo, p, length);
		break;

	default:
		if (!ndo->ndo_eflag)
			nflog_hdr_print(ndo, hdr,
					length + NFLOG_HDR_LEN);

		if (!ndo->ndo_suppress_default_print)
			ND_DEFAULTPRINT(p, caplen);
		break;
	}

	ndo->ndo_ll_hdr_len += h_size - NFLOG_HDR_LEN;
	return;
trunc:
	nd_print_trunc(ndo);
	ndo->ndo_ll_hdr_len += h_size - NFLOG_HDR_LEN;
}

#endif /* DLT_NFLOG */
