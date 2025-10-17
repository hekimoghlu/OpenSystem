/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 19, 2024.
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
#ifndef __PPPOE_H__
#define __PPPOE_H__

#include <sys/ioccom.h>
#include <net/ethernet.h>

// PPPoE error codes (bits 8..15 of last cause key)
#define EXIT_PPPoE_NOSERVER  		1
#define EXIT_PPPoE_NOSERVICE  		2
#define EXIT_PPPoE_NOAC 		3
#define EXIT_PPPoE_NOACSERVICE 		4
#define EXIT_PPPoE_CONNREFUSED 		5

#define PPPOE_ETHERTYPE_CTRL 	0x8863
#define PPPOE_ETHERTYPE_DATA 	0x8864

//#define PF_PPPOE 		247		/* TEMP - move to socket.h */
#define PPPPROTO_PPPOE		16		/* TEMP - move to ppp.h - 1..32 are reserved */
//#define APPLE_PPP_NAME_PPPoE	"PPPoE"
#define PPPOE_NAME		"PPPoE"		/* */

#define PPPOE_AC_NAME_LEN	64
#define PPPOE_SERVICE_LEN	64

struct sockaddr_pppoe
{
    struct sockaddr_ppp	ppp;					/* generic ppp address */
    char 		pppoe_ac_name[PPPOE_AC_NAME_LEN];	/* Access Concentrator name */
    char 		pppoe_service[PPPOE_SERVICE_LEN];	/* Service name */
};


#define PPPOE_OPT_FLAGS		1	/* see flags definition below */
#define PPPOE_OPT_INTERFACE	2	/* ethernet interface to use (en0, en1...) */
#define PPPOE_OPT_CONNECT_TIMER	3	/* time allowed for outgoing call (in seconds) */
#define PPPOE_OPT_RING_TIMER	4	/* time allowed for incoming call (in seconds) */
#define PPPOE_OPT_RETRY_TIMER	5	/* connection retry timer (in seconds) */
#define PPPOE_OPT_PEER_ENETADDR	6	/* peer ethernet address */

/* flags definition */
#define PPPOE_FLAG_LOOPBACK	0x00000001	/* loopback mode, for debugging purpose */
#define PPPOE_FLAG_DEBUG	0x00000002	/* debug mode, send verbose logs to syslog */
#define PPPOE_FLAG_PROBE	0x00000004	/* just probe to detect presence of servers */


#endif
