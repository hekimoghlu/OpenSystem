/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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
/* \summary: - Locator/Identifier Separation Protocol (LISP) printer */

/*
 * specification: RFC 6830
 *
 *
 * The Map-Register message format is:
 *
 *       0                   1                   2                   3
 *       0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 *      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *      |Type=3 |P|S|I|R|      Reserved               |M| Record Count  |
 *      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *      |                         Nonce . . .                           |
 *      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *      |                         . . . Nonce                           |
 *      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *      |            Key ID             |  Authentication Data Length   |
 *      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *      ~                     Authentication Data                       ~
 *  +-> +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  |   |                          Record TTL                           |
 *  |   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  R   | Locator Count | EID mask-len  | ACT |A|      Reserved         |
 *  e   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  c   | Rsvd  |  Map-Version Number   |        EID-Prefix-AFI         |
 *  o   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  r   |                          EID-Prefix                           |
 *  d   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  |  /|    Priority   |    Weight     |  M Priority   |   M Weight    |
 *  | L +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  | o |        Unused Flags     |L|p|R|           Loc-AFI             |
 *  | c +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  |  \|                             Locator                           |
 *  +-> +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *
 *
 * The Map-Notify message format is:
 *
 *       0                   1                   2                   3
 *       0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 *      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *      |Type=4 |I|R|          Reserved                 | Record Count  |
 *      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *      |                         Nonce . . .                           |
 *      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *      |                         . . . Nonce                           |
 *      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *      |            Key ID             |  Authentication Data Length   |
 *      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *      ~                     Authentication Data                       ~
 *  +-> +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  |   |                          Record TTL                           |
 *  |   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  R   | Locator Count | EID mask-len  | ACT |A|      Reserved         |
 *  e   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  c   | Rsvd  |  Map-Version Number   |         EID-Prefix-AFI        |
 *  o   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  r   |                          EID-Prefix                           |
 *  d   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  |  /|    Priority   |    Weight     |  M Priority   |   M Weight    |
 *  | L +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  | o |        Unused Flags     |L|p|R|           Loc-AFI             |
 *  | c +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 *  |  \|                             Locator                           |
 *  +-> +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"
#include "netdissect.h"

#include "ip.h"
#include "ip6.h"

#include "extract.h"
#include "addrtoname.h"


#define IPv4_AFI			1
#define IPv6_AFI			2
#define TYPE_INDEX			4
#define LISP_MAP_NOTIFY_IBIT_MASK	8
#define LISP_MAP_REGISTER_IBIT_MASK	2

enum {
	LISP_MAP_REQUEST = 1,
	LISP_MAP_REPLY,
	LISP_MAP_REGISTER,
	LISP_MAP_NOTIFY,
	LISP_ENCAPSULATED_CONTROL_MESSAGE = 8
};

enum {
	LISP_AUTH_NONE,
	LISP_AUTH_SHA1,
	LISP_AUTH_SHA256
};

static const struct tok lisp_type [] = {
	{ 0, "LISP-Reserved"			},
	{ 1, "LISP-Map-Request"			},
	{ 2, "LISP-Map-Reply"			},
	{ 3, "LISP-Map-Register"		},
	{ 4, "LISP-Map-Notify"			},
	{ 8, "LISP-Encapsulated-Contol-Message" },
	{ 0, NULL }
};

/*
 * P-Bit : Request for Proxy Map-Reply from the MS/MR
 * S-Bit : Security Enhancement. ETR is LISP-SEC enabled. draft-ietf-lisp-sec
 * I-Bit : 128 bit xTR-ID and 64 bit Site-ID present.
 *	   xTR-ID and Site-ID help in differentiation of xTRs in multi xTR
 *	   and multi Site deployment scenarios.
 * R-Bit : Built for a Reencapsulating-Tunnel-Router. Used in Traffic
 *	   Engineering and Service Chaining
 */
static const struct tok map_register_hdr_flag[] = {
	{ 0x08000000, "P-Proxy-Map-Reply"  },
	{ 0x04000000, "S-LISP-SEC-Capable" },
	{ 0x02000000, "I-xTR-ID-Present"   },
	{ 0x01000000, "R-Build-For-RTR"    },
	{ 0x00000100, "M-Want-Map-Notify"  },
	{ 0, NULL }
};

static const struct tok map_notify_hdr_flag[] = {
	{ 0x08000000, "I-xTR-ID-Present"   },
	{ 0x04000000, "R-Build-For-RTR"    },
	{ 0, NULL }
};

static const struct tok auth_type[] = {
	{ LISP_AUTH_NONE,   "None"   },
	{ LISP_AUTH_SHA1,   "SHA1"   },
	{ LISP_AUTH_SHA256, "SHA256" },
	{ 0, NULL}
};

static const struct tok lisp_eid_action[] = {
	{ 0, "No-Action"	},
	{ 1, "Natively-Forward" },
	{ 2, "Send-Map-Request" },
	{ 3, "Drop"		},
	{ 0, NULL}
};

static const struct tok lisp_loc_flag[] = {
	{ 0x0004, "Local-Locator" },
	{ 0x0002, "RLoc-Probed"	  },
	{ 0x0001, "Reachable"	  },
	{ 0, NULL }
};

typedef struct map_register_hdr {
	nd_uint8_t type_and_flag;
	nd_uint8_t reserved;
	nd_uint8_t reserved_and_flag2;
	nd_uint8_t record_count;
	nd_uint64_t nonce;
	nd_uint16_t key_id;
	nd_uint16_t auth_data_len;
} lisp_map_register_hdr;

#define MAP_REGISTER_HDR_LEN sizeof(lisp_map_register_hdr)

typedef struct map_register_eid {
	nd_uint32_t ttl;
	nd_uint8_t locator_count;
	nd_uint8_t eid_prefix_mask_length;
	nd_uint8_t act_auth_inc_res;
	nd_uint8_t reserved;
	nd_uint16_t reserved_and_version;
	nd_uint16_t eid_prefix_afi;
} lisp_map_register_eid;

#define MAP_REGISTER_EID_LEN sizeof(lisp_map_register_eid)

typedef struct map_register_loc {
	nd_uint8_t priority;
	nd_uint8_t weight;
	nd_uint8_t m_priority;
	nd_uint8_t m_weight;
	nd_uint16_t unused_and_flag;
	nd_uint16_t locator_afi;
} lisp_map_register_loc;

#define MAP_REGISTER_LOC_LEN sizeof(lisp_map_register_loc)

static uint8_t extract_lisp_type(uint8_t);
static uint8_t is_xtr_data_present(uint8_t, uint8_t);
static void lisp_hdr_flag(netdissect_options *, const lisp_map_register_hdr *);
static void action_flag(netdissect_options *, uint8_t);
static void loc_hdr_flag(netdissect_options *, uint16_t);

void
lisp_print(netdissect_options *ndo, const u_char *bp, u_int length)
{
	uint8_t type_and_flag;
	uint8_t type;
	uint8_t mask_len;
	uint8_t loc_count;
	uint8_t xtr_present;
	uint8_t record_count;
	uint16_t key_id;
	uint16_t eid_afi;
	uint16_t loc_afi;
	uint16_t map_version;
	uint16_t packet_offset;
	uint16_t auth_data_len;
	uint32_t ttl;
	const u_char *packet_iterator;
	const u_char *loc_ip_pointer;
	const lisp_map_register_hdr *lisp_hdr;
	const lisp_map_register_eid *lisp_eid;
	const lisp_map_register_loc *lisp_loc;

	ndo->ndo_protocol = "lisp";
	/* Check if enough bytes for header are available */
	ND_TCHECK_LEN(bp, MAP_REGISTER_HDR_LEN);
	lisp_hdr = (const lisp_map_register_hdr *) bp;
	lisp_hdr_flag(ndo, lisp_hdr);
	/* Supporting only MAP NOTIFY and MAP REGISTER LISP packets */
	type_and_flag = GET_U_1(lisp_hdr->type_and_flag);
	type = extract_lisp_type(type_and_flag);
	if ((type != LISP_MAP_REGISTER) && (type != LISP_MAP_NOTIFY))
		return;

	/* Find if the packet contains xTR and Site-ID data */
	xtr_present = is_xtr_data_present(type, type_and_flag);

	/* Extract the number of EID records present */
	auth_data_len = GET_BE_U_2(lisp_hdr->auth_data_len);
	packet_iterator = (const u_char *)(lisp_hdr);
	packet_offset = MAP_REGISTER_HDR_LEN;
	record_count = GET_U_1(lisp_hdr->record_count);

	if (ndo->ndo_vflag) {
		key_id = GET_BE_U_2(lisp_hdr->key_id);
		ND_PRINT("\n    %u record(s), ", record_count);
		ND_PRINT("Authentication %s,",
			tok2str(auth_type, "unknown-type", key_id));
		hex_print(ndo, "\n    Authentication-Data: ", packet_iterator +
						packet_offset, auth_data_len);
	} else {
		ND_PRINT(" %u record(s),", record_count);
	}
	packet_offset += auth_data_len;

	if (record_count == 0)
		goto invalid;

	/* Print all the EID records */
	while ((length > packet_offset) && (record_count--)) {

		ND_TCHECK_LEN(packet_iterator + packet_offset,
			      MAP_REGISTER_EID_LEN);
		ND_PRINT("\n");
		lisp_eid = (const lisp_map_register_eid *)
				((const u_char *)lisp_hdr + packet_offset);
		packet_offset += MAP_REGISTER_EID_LEN;
		mask_len = GET_U_1(lisp_eid->eid_prefix_mask_length);
		eid_afi = GET_BE_U_2(lisp_eid->eid_prefix_afi);
		loc_count = GET_U_1(lisp_eid->locator_count);

		if (ndo->ndo_vflag) {
			ttl = GET_BE_U_4(lisp_eid->ttl);
			ND_PRINT("      Record TTL %u,", ttl);
			action_flag(ndo, GET_U_1(lisp_eid->act_auth_inc_res));
			map_version = GET_BE_U_2(lisp_eid->reserved_and_version) & 0x0FFF;
			ND_PRINT(" Map Version: %u,", map_version);
		}

		switch (eid_afi) {
		case IPv4_AFI:
			ND_PRINT(" EID %s/%u,",
				GET_IPADDR_STRING(packet_iterator + packet_offset),
				mask_len);
			packet_offset += 4;
			break;
		case IPv6_AFI:
			ND_PRINT(" EID %s/%u,",
				GET_IP6ADDR_STRING(packet_iterator + packet_offset),
				mask_len);
			packet_offset += 16;
			break;
		default:
			/*
			 * No support for LCAF right now.
			 */
			return;
			break;
		}

		ND_PRINT(" %u locator(s)", loc_count);

		while (loc_count--) {
			ND_TCHECK_LEN(packet_iterator + packet_offset,
				      MAP_REGISTER_LOC_LEN);
			lisp_loc = (const lisp_map_register_loc *) (packet_iterator + packet_offset);
			loc_ip_pointer = (const u_char *) (lisp_loc + 1);
			packet_offset += MAP_REGISTER_LOC_LEN;
			loc_afi = GET_BE_U_2(lisp_loc->locator_afi);

			if (ndo->ndo_vflag)
				ND_PRINT("\n       ");

			switch (loc_afi) {
			case IPv4_AFI:
				ND_TCHECK_4(packet_iterator + packet_offset);
				ND_PRINT(" LOC %s", GET_IPADDR_STRING(loc_ip_pointer));
				packet_offset += 4;
				break;
			case IPv6_AFI:
				ND_TCHECK_16(packet_iterator + packet_offset);
				ND_PRINT(" LOC %s", GET_IP6ADDR_STRING(loc_ip_pointer));
				packet_offset += 16;
				break;
			default:
				break;
			}
			if (ndo->ndo_vflag) {
				ND_PRINT("\n          Priority/Weight %u/%u,"
						" Multicast Priority/Weight %u/%u,",
						GET_U_1(lisp_loc->priority),
						GET_U_1(lisp_loc->weight),
						GET_U_1(lisp_loc->m_priority),
						GET_U_1(lisp_loc->m_weight));
				loc_hdr_flag(ndo,
					     GET_BE_U_2(lisp_loc->unused_and_flag));
			}
		}
	}

	/*
	 * Print xTR and Site ID. Handle the fact that the packet could be invalid.
	 * If the xTR_ID_Present bit is not set, and we still have data to display,
	 * show it as hex data.
	 */
	if (xtr_present) {
		if (!ND_TTEST_LEN(packet_iterator + packet_offset, 24))
			goto invalid;
		hex_print(ndo, "\n    xTR-ID: ", packet_iterator + packet_offset, 16);
		ND_PRINT("\n    SITE-ID: %" PRIu64,
			GET_BE_U_8(packet_iterator + packet_offset + 16));
	} else {
		/* Check if packet isn't over yet */
		if (packet_iterator + packet_offset < ndo->ndo_snapend) {
			hex_print(ndo, "\n    Data: ", packet_iterator + packet_offset,
				ND_BYTES_AVAILABLE_AFTER(packet_iterator + packet_offset));
		}
	}
	return;
trunc:
	nd_print_trunc(ndo);
	return;
invalid:
	nd_print_invalid(ndo);
}

static uint8_t
extract_lisp_type(uint8_t lisp_hdr_flags)
{
	return (lisp_hdr_flags) >> TYPE_INDEX;
}

static uint8_t
is_xtr_data_present(uint8_t type, uint8_t lisp_hdr_flags)
{
	uint8_t xtr_present = 0;

	if (type == LISP_MAP_REGISTER)
		xtr_present = (lisp_hdr_flags) & LISP_MAP_REGISTER_IBIT_MASK;
	else if (type == LISP_MAP_NOTIFY)
		xtr_present = (lisp_hdr_flags) & LISP_MAP_NOTIFY_IBIT_MASK;

	return xtr_present;
}

static void lisp_hdr_flag(netdissect_options *ndo, const lisp_map_register_hdr *lisp_hdr)
{
	uint8_t type = extract_lisp_type(GET_U_1(lisp_hdr->type_and_flag));

	if (!ndo->ndo_vflag) {
		ND_PRINT("%s,", tok2str(lisp_type, "unknown-type-%u", type));
		return;
	} else {
		ND_PRINT("%s,", tok2str(lisp_type, "unknown-type-%u", type));
	}

	if (type == LISP_MAP_REGISTER) {
		ND_PRINT(" flags [%s],", bittok2str(map_register_hdr_flag,
			 "none", GET_BE_U_4(lisp_hdr)));
	} else if (type == LISP_MAP_NOTIFY) {
		ND_PRINT(" flags [%s],", bittok2str(map_notify_hdr_flag,
			 "none", GET_BE_U_4(lisp_hdr)));
	}
}

static void action_flag(netdissect_options *ndo, uint8_t act_auth_inc_res)
{
	uint8_t action;
	uint8_t authoritative;

	authoritative  = ((act_auth_inc_res >> 4) & 1);

	if (authoritative)
		ND_PRINT(" Authoritative,");
	else
		ND_PRINT(" Non-Authoritative,");

	action = act_auth_inc_res >> 5;
	ND_PRINT(" %s,", tok2str(lisp_eid_action, "unknown", action));
}

static void loc_hdr_flag(netdissect_options *ndo, uint16_t flag)
{
	ND_PRINT(" flags [%s],", bittok2str(lisp_loc_flag, "none", flag));
}

