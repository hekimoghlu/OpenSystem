/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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
 * sysconfig.h
 * - system configuration related functions
 */

/* 
 * Modification History
 *
 * June 23, 2009	Dieter Siegmund (dieter@apple.com)
 * - split out from ipconfigd.c
 */

#ifndef _S_SYSCONFIG_H
#define _S_SYSCONFIG_H

#include <CoreFoundation/CFDictionary.h>
#include <CoreFoundation/CFString.h>
#include <SystemConfiguration/SCDynamicStore.h>
#include <stdint.h>
#include "timer.h"
#include "ipconfigd_types.h"
#include "dhcp_options.h"
#include "DHCPv6Options.h"

/*
 * Define: kServiceEntity
 * Purpose:
 *   Placeholder entity key for the service dictionary.
 */
#define kServiceEntity		kCFNull

CFDictionaryRef
my_SCDynamicStoreCopyDictionary(SCDynamicStoreRef session, CFStringRef key);

/*
 * Function: my_SCDynamicStoreSetService
 * Purpose:
 *   Accumulate the keys to set/remove for a particular service.
 * Note:
 *   This function does not update the SCDynamicStore, it just 
 *   accumulates keys/values.
 */
void
my_SCDynamicStoreSetService(SCDynamicStoreRef store,
			    CFStringRef serviceID,
			    CFTypeRef * entities,
			    CFDictionaryRef * values,
			    int count,
			    boolean_t alternate_location);

/*
 * Function: my_SCDynamicStoreSetInterface
 * Purpose:
 *   Accumulate the keys to set/remove for a particular interface.
 * Note:
 *   This function does not update the SCDynamicStore, it just
 *   accumulates keys/values.
 */
void
my_SCDynamicStoreSetInterface(SCDynamicStoreRef store,
			      CFStringRef ifname,
			      CFStringRef entity,
			      CFDictionaryRef value);

/*
 * Function: my_SCDynamicStorePublish
 * Purpose:
 *   Update the SCDynamicStore with the accumulated keys/values generated
 *   by previous calls to my_SCDynamicStoreSetService().
 */
void
my_SCDynamicStorePublish(SCDynamicStoreRef store);
				
CFDictionaryRef
DHCPInfoDictionaryCreate(ipconfig_method_t method, dhcpol_t * options_p,
			 absolute_time_t start_time,
			 absolute_time_t expiration_time);

CFDictionaryRef
DHCPv6InfoDictionaryCreate(ipv6_info_t * info_p);

CFStringRef
IPv4ARPCollisionKeyParse(CFStringRef cache_key, struct in_addr * ipaddr_p,
			 void * * hwaddr, int * hwlen);

CFDictionaryRef
DNSEntityCreateWithInfo(const char * if_name,
			dhcp_info_t * info_v4_p,
			ipv6_info_t * info_v6_p);

CFDictionaryRef
CaptivePortalEntityCreateWithInfo(dhcp_info_t * info_v4_p,
				  ipv6_info_t * info_v6_p);

void *
bytesFromColonHexString(CFStringRef colon_hex, int * len);

CFDictionaryRef
route_dict_create(const struct in_addr * dest, const struct in_addr * mask,
		  const struct in_addr * gate);

CFDictionaryRef
PvDEntityCreateWithInfo(ipv6_info_t * info_p);

#endif /* _S_SYSCONFIG_H */
