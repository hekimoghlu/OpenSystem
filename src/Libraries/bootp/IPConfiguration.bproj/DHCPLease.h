/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
 * DHCPLease.h
 * - in-memory and persistent DHCP lease support
 */

#ifndef _S_DHCPLEASE_H
#define _S_DHCPLEASE_H

/* 
 * Modification History
 *
 * June 11, 2009		Dieter Siegmund (dieter@apple.com)
 * - split out from ipconfigd.c
 */

#include <stdbool.h>
#include <stdint.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include "dhcplib.h"
#include "timer.h"
#include "interfaces.h"
#include "arp_session.h"

#define DHCP_LEASE_NOT_FOUND	(-1)

/**
 ** DHCPLease, DHCPLeaseList
 **/
typedef struct {
    bool			tentative;
    bool			nak;
    struct in_addr		our_ip;
    absolute_time_t		lease_start;
    dhcp_lease_time_t		lease_length;
    struct in_addr		router_ip;
    uint8_t			router_hwaddr[MAX_LINK_ADDR_LEN];
    uint8_t			router_hwaddr_length;
    uint8_t			wifi_mac[ETHER_ADDR_LEN];
    bool			wifi_mac_is_set;
    CFStringRef			ssid;
    CFStringRef			networkID;
    int				pkt_length;
    uint8_t			pkt[1];
} DHCPLease, * DHCPLeaseRef;

typedef dynarray_t DHCPLeaseList, * DHCPLeaseListRef;

void
DHCPLeaseSetNAK(DHCPLeaseRef lease_p, int nak);


void
DHCPLeaseListInit(DHCPLeaseListRef list_p);

void
DHCPLeaseListFree(DHCPLeaseListRef list_p);

void
DHCPLeaseListRemoveLease(DHCPLeaseListRef list_p,
			 struct in_addr our_ip,
			 struct in_addr router_ip,
			 const uint8_t * router_hwaddr,
			 int router_hwaddr_length);
void
DHCPLeaseListUpdateLease(DHCPLeaseListRef list_p,
			 struct in_addr our_ip,
			 struct in_addr router_ip,
			 const uint8_t * router_hwaddr,
			 int router_hwaddr_length,
			 absolute_time_t lease_start,
			 dhcp_lease_time_t lease_length,
			 const uint8_t * pkt, int pkt_length,
			 CFStringRef ssid, CFStringRef networkID);
arp_address_info_t *
DHCPLeaseListCopyARPAddressInfo(DHCPLeaseListRef list_p,
				CFStringRef ssid,
				CFStringRef networkID,
				absolute_time_t * start_threshold_p,
				bool tentative_ok,
				int * ret_count);


void
DHCPLeaseListWrite(DHCPLeaseListRef list_p,
		   const char * ifname,
		   uint8_t cid_type, const void * cid, int cid_length);
void
DHCPLeaseListRead(DHCPLeaseListRef list_p,
		  const char * ifname, bool is_wifi,
		  uint8_t cid_type, const void * cid, int cid_length);

int
DHCPLeaseListFindLease(DHCPLeaseListRef list_p, struct in_addr our_ip,
		       struct in_addr router_ip,
		       const uint8_t * router_hwaddr, int router_hwaddr_length);

int
DHCPLeaseListFindLeaseForWiFi(DHCPLeaseListRef list_p, CFStringRef ssid,
			      CFStringRef networkID);

void
DHCPLeaseListRemoveLeaseForWiFi(DHCPLeaseListRef list_p, CFStringRef ssid,
				CFStringRef networkID);

static __inline__ int
DHCPLeaseListCount(DHCPLeaseListRef list_p)
{
    return (dynarray_count(list_p));
}

static __inline__ DHCPLeaseRef
DHCPLeaseListElement(DHCPLeaseListRef list_p, int i)
{
    return (dynarray_element(list_p, i));
}

#endif /* _S_DHCPLEASE_H */
