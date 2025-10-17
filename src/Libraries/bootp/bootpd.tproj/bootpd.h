/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#ifndef _S_BOOTPD_H
#define _S_BOOTPD_H

#include "dhcp_options.h"
#include <CoreFoundation/CFDictionary.h>
#include <CoreFoundation/CFString.h>
#include "netinfo.h"
#include "mylog.h"

#if TARGET_OS_IPHONE

#define NETBOOT_SERVER_SUPPORT	0
#define USE_OPEN_DIRECTORY	0

#else /* TARGET_OS_IPHONE */

#if MINIMAL_DHCP_SERVER
#define NETBOOT_SERVER_SUPPORT	0
#define USE_OPEN_DIRECTORY	0
#else /* MINIMAL_DHCP_SERVER */
#define NETBOOT_SERVER_SUPPORT	1
#define USE_OPEN_DIRECTORY	1
#endif /* MINIMAL_DHCP_SERVER*/

#endif /* TARGET_OS_IPHONE */


typedef struct {
    interface_t *	if_p;
    struct dhcp *	pkt;
    int			pkt_length;
    dhcpol_t *		options_p;
    struct in_addr *	dstaddr_p;
    struct timeval *	time_in_p;
} request_t;

/*
 * bootpd.h
 */
int
add_subnet_options(char * hostname, 
		   struct in_addr iaddr, 
		   interface_t * intface, dhcpoa_t * options,
		   const uint8_t * tags, int n);
boolean_t
bootp_add_bootfile(const char * request_file, const char * hostname, 
		   const char * bootfile, char * reply_file,
		   int reply_file_size);
void
host_parms_from_proplist(ni_proplist * pl_p, int index, struct in_addr * ip, 
			 char * * name, char * * bootfile);

boolean_t
subnetAddressAndMask(struct in_addr giaddr, interface_t * intface,
		     struct in_addr * addr, struct in_addr * mask);
boolean_t
subnet_match(void * arg, struct in_addr iaddr);

boolean_t	
sendreply(interface_t * intf, struct bootp * bp, int n,
	  boolean_t broadcast, struct in_addr * dest_p);
boolean_t
ip_address_reachable(struct in_addr ip, struct in_addr giaddr, 
		     interface_t * intface);

void
set_number_from_plist(CFDictionaryRef plist, CFStringRef prop_name_cf,
		      const char * prop_name, uint32_t * val_p);

#define NI_DHCP_OPTION_PREFIX	"dhcp_"
#include "globals.h"

typedef struct subnet_match_args {
    struct in_addr	giaddr;
    struct in_addr	ciaddr;
    interface_t *	if_p;
    boolean_t		has_binding;
} subnet_match_args_t;

boolean_t
detect_other_dhcp_server(interface_t * if_p);

boolean_t
ipv6_only_preferred(interface_t * if_p);

void
disable_dhcp_on_interface(interface_t * if_p);

#endif /* _S_BOOTPD_H */
