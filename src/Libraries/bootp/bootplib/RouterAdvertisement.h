/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
 * RouterAdvertisement.h
 * - CF object to encapulate an IPv6 ND Router Advertisement
 */

/*
 * Modification History
 *
 * April 15, 2020		Dieter Siegmund (dieter@apple.com)
 * - created
 */

#ifndef _S_ROUTERADVERTISEMENT_H
#define _S_ROUTERADVERTISEMENT_H

#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/ip6.h>
#include <netinet/icmp6.h>
#include <CoreFoundation/CFDate.h>
#include <CoreFoundation/CFString.h>
#include "symbol_scope.h"
#include "ipconfig_types.h"

#ifndef ND_OPT_ALIGN
#define ND_OPT_ALIGN			8
#endif /* ND_OPT_ALIGN */

#ifndef ND_RA_FLAG_PROXY
#define ND_RA_FLAG_PROXY	0x04
#endif /* ND_RA_FLAGS_PROXY */

#define ROUTER_LIFETIME_MAXIMUM		((uint16_t)0xffff)

typedef struct __RouterAdvertisement * RouterAdvertisementRef;

typedef struct {
	bool            http;
	bool            legacy;
	bool            ra;
	uint16_t        delay;
} RA_PvDFlagsDelay, * RA_PvDFlagsDelayRef;

RouterAdvertisementRef
RouterAdvertisementCreate(const struct nd_router_advert * ndra,
			  size_t ndra_length, const struct in6_addr * from,
			  CFAbsoluteTime receive_time);

CFAbsoluteTime
RouterAdvertisementGetReceiveTime(RouterAdvertisementRef ra);

bool
RouterAdvertisementLifetimeHasExpired(RouterAdvertisementRef ra,
				      CFAbsoluteTime now,
				      uint32_t lifetime);
CFStringRef
RouterAdvertisementGetSourceIPAddressAsString(RouterAdvertisementRef ra);

const struct in6_addr *
RouterAdvertisementGetSourceIPAddress(RouterAdvertisementRef ra);

CFStringRef
RouterAdvertisementCopyDescription(RouterAdvertisementRef ra);

uint16_t
RouterAdvertisementGetRouterLifetime(RouterAdvertisementRef ra);

const uint8_t *
RouterAdvertisementGetSourceLinkAddress(RouterAdvertisementRef ra,
					int * ret_len);

CFArrayRef
RouterAdvertisementCopyPrefixes(RouterAdvertisementRef ra);

uint32_t
RouterAdvertisementGetPrefixLifetimes(RouterAdvertisementRef ra,
				      uint32_t * valid_lifetime);

uint8_t
RouterAdvertisementGetFlags(RouterAdvertisementRef ra);

INLINE bool
RouterAdvertisementFlagsGetIsManaged(RouterAdvertisementRef ra)
{
	uint8_t	flags = RouterAdvertisementGetFlags(ra);
	return ((flags & ND_RA_FLAG_MANAGED) != 0);
}

INLINE bool
RouterAdvertisementFlagsGetIsOther(RouterAdvertisementRef ra)
{
	uint8_t	flags = RouterAdvertisementGetFlags(ra);
	return ((flags & ND_RA_FLAG_OTHER) != 0);
}

const struct in6_addr *
RouterAdvertisementGetRDNSS(RouterAdvertisementRef ra,
			    int * dns_servers_count_p,
			    uint32_t * lifetime_p);

const uint8_t *
RouterAdvertisementGetDNSSL(RouterAdvertisementRef ra,
			    int * domains_length_p,
			    uint32_t * lifetime_p);

const uint8_t *
RouterAdvertisementGetPvD(RouterAdvertisementRef ra,
                          size_t * pvd_id_length,
                          uint16_t * sequence,
                          RA_PvDFlagsDelayRef flags);

CFAbsoluteTime
RouterAdvertisementGetDNSExpirationTime(RouterAdvertisementRef ra,
					CFAbsoluteTime now,
					bool * has_dns,
					bool * has_expired);

CFStringRef
RouterAdvertisementCopyCaptivePortal(RouterAdvertisementRef ra);

bool
RouterAdvertisementGetPREF64(RouterAdvertisementRef ra,
			     struct in6_addr * ret_prefix,
			     uint8_t * ret_prefix_length,
			     uint16_t * ret_prefix_lifetime);

CFStringRef
RouterAdvertisementCopyPREF64PrefixAndLifetime(RouterAdvertisementRef ra,
					       uint16_t * ret_prefix_lifetime);

RouterAdvertisementRef
RouterAdvertisementCreateWithDictionary(CFDictionaryRef dict);

CFDictionaryRef
RouterAdvertisementCopyDictionary(RouterAdvertisementRef ra);

#endif /* _S_ROUTERADVERTISEMENT_H */
