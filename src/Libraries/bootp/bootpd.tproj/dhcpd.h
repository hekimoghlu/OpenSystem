/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
 * dhcpd.h
 * - DHCP server definitions
 */

#ifndef _S_DHCPD_H
#define _S_DHCPD_H


#include "dhcplib.h"
#include "bootpd.h"

void
dhcp_init();

void
dhcp_request(request_t * request, dhcp_msgtype_t msgtype,
	     boolean_t dhcp_allocate);

boolean_t
dhcp_bootp_allocate(char * idstr, char * hwstr, struct dhcp * rq,
		    interface_t * if_p, struct timeval * time_in_p,
		    struct in_addr * iaddr_p, SubnetRef * subnet_p);

#define DHCP_CLIENT_TYPE		"dhcp"

/* default time to leave an ip address pending before re-using it */
#define DHCP_PENDING_SECS		60

#define DHCP_DECLINE_WAIT_SECS (60 * 10)		/* 10 minutes */

struct dhcp * 
make_dhcp_reply(struct dhcp * reply, int pkt_size, 
		struct in_addr server_id, dhcp_msgtype_t msg, 
		struct dhcp * request, dhcpoa_t * options);
int
dhcp_max_message_size(dhcpol_t * client_options);

#endif /* _S_DHCPD_H */
