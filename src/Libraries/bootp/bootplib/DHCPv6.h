/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
/*
 * DHCPv6.h
 * - definitions for DHCPv6
 */

/* 
 * Modification History
 *
 * September 16, 2009		Dieter Siegmund (dieter@apple.com)
 * - created
 */


#ifndef _S_DHCPV6_H
#define _S_DHCPV6_H

#include <stdint.h>
#include <stdio.h>
#include <netinet/in.h>
#include <stddef.h>
#include <CoreFoundation/CFString.h>
#include "symbol_scope.h"
#include "DHCPDUID.h"
#include "nbo.h"

#define DHCPV6_CLIENT_PORT	546
#define DHCPV6_SERVER_PORT	547

/*
 * Constant: kAll_DHCP_Relay_Agents_and_Servers (FF02::1:2) 
 */
#define All_DHCP_Relay_Agents_and_Servers_INIT \
    {{{ 0xff, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, \
        0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02  }}}

typedef CF_ENUM(uint8_t, DHCPv6MessageType) {
    kDHCPv6MessageNone 			= 0,
    kDHCPv6MessageSOLICIT		= 1,
    kDHCPv6MessageADVERTISE		= 2,
    kDHCPv6MessageREQUEST		= 3,
    kDHCPv6MessageCONFIRM		= 4,
    kDHCPv6MessageRENEW			= 5,
    kDHCPv6MessageREBIND		= 6,
    kDHCPv6MessageREPLY			= 7,
    kDHCPv6MessageRELEASE		= 8,
    kDHCPv6MessageDECLINE		= 9,
    kDHCPv6MessageRECONFIGURE		= 10,
    kDHCPv6MessageINFORMATION_REQUEST	= 11,
    kDHCPv6MessageRELAY_FORW		= 12,
    kDHCPv6MessageRELAY_REPL		= 13
};

/* 
 * Transmission and re-transmission timers
 */
#define DHCPv6_SOL_MAX_DELAY     1 /* Max delay of first Solicit */
#define DHCPv6_SOL_TIMEOUT       1 /* Initial Solicit timeout */
#define DHCPv6_SOL_MAX_RT     3600 /* Max Solicit timeout value */
#define DHCPv6_REQ_TIMEOUT       1 /* Initial Request timeout */
#define DHCPv6_REQ_MAX_RT       30 /* Max Request timeout value */
#define DHCPv6_REQ_MAX_RC       10 /* Max Request retry attempts */
#define DHCPv6_CNF_MAX_DELAY     1 /* Max delay of first Confirm */
#define DHCPv6_CNF_TIMEOUT       1 /* Initial Confirm timeout */
#define DHCPv6_CNF_MAX_RT        4 /* Max Confirm timeout */
#define DHCPv6_CNF_MAX_RD       10 /* Max Confirm duration */
#define DHCPv6_REN_TIMEOUT      10 /* Initial Renew timeout */
#define DHCPv6_REN_MAX_RT      600 /* Max Renew timeout value */
#define DHCPv6_REB_TIMEOUT      10 /* Initial Rebind timeout */
#define DHCPv6_REB_MAX_RT      600 /* Max Rebind timeout value */
#define DHCPv6_INF_MAX_DELAY     1 /* Max delay of first Information-request */
#define DHCPv6_INF_TIMEOUT       1 /* Initial Information-request timeout */
#define DHCPv6_INF_MAX_RT     3600 /* Max Information-request timeout value */
#define DHCPv6_REL_TIMEOUT       1 /* Initial Release timeout */
#define DHCPv6_REL_MAX_RC        5 /* MAX Release attempts */
#define DHCPv6_DEC_TIMEOUT       1 /* Initial Decline timeout */
#define DHCPv6_DEC_MAX_RC        5 /* Max Decline attempts */
#define DHCPv6_REC_TIMEOUT       2 /* Initial Reconfigure timeout */
#define DHCPv6_REC_MAX_RC        8 /* Max Reconfigure attempts */
#define DHCPv6_HOP_COUNT_LIMIT  32 /* Max hop count in Relay-forward message */

const char *
DHCPv6MessageTypeName(DHCPv6MessageType msg_type);

/**
 ** DHCPv6 Packet
 **/
typedef struct {
    uint8_t		msg_type;
    uint8_t		transaction_id[3];
    uint8_t		options[1]; /* variable length */
} DHCPv6Packet, * DHCPv6PacketRef;

#define DHCPV6_PACKET_HEADER_LENGTH	((int)offsetof(DHCPv6Packet, options))

typedef uint32_t	DHCPv6TransactionID;

DHCPv6TransactionID
DHCPv6PacketGetTransactionID(const DHCPv6PacketRef pkt);

void
DHCPv6PacketSetTransactionID(DHCPv6PacketRef pkt,
			     DHCPv6TransactionID transaction_id);

void
DHCPv6PacketSetMessageType(DHCPv6PacketRef pkt, DHCPv6MessageType msg_type);

void
DHCPv6PacketFPrint(FILE * file, const DHCPv6PacketRef pkt, int pkt_len);

void
DHCPv6PacketPrintToString(CFMutableStringRef str,
			  const DHCPv6PacketRef pkt, int pkt_len);

#endif /* _S_DHCPV6_H */
