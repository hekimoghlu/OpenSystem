/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
#ifndef lib_pcap_nflog_h
#define lib_pcap_nflog_h

#include <pcap/pcap-inttypes.h>

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
	uint8_t		nflog_family;	/* address family */
	uint8_t		nflog_version;	/* version */
	uint16_t	nflog_rid;	/* resource ID */
} nflog_hdr_t;

typedef struct nflog_tlv {
	uint16_t	tlv_length;	/* tlv length */
	uint16_t	tlv_type;	/* tlv type */
	/* value follows this */
} nflog_tlv_t;

typedef struct nflog_packet_hdr {
	uint16_t	hw_protocol;	/* hw protocol */
	uint8_t		hook;		/* netfilter hook */
	uint8_t		pad;		/* padding to 32 bits */
} nflog_packet_hdr_t;

typedef struct nflog_hwaddr {
	uint16_t	hw_addrlen;	/* address length */
	uint16_t	pad;		/* padding to 32-bit boundary */
	uint8_t		hw_addr[8];	/* address, up to 8 bytes */
} nflog_hwaddr_t;

typedef struct nflog_timestamp {
	uint64_t	sec;
	uint64_t	usec;
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

#endif
