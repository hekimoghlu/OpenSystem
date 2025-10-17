/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
#ifndef _IPMONITOR_CONTROL_PRIVATE_H
#define _IPMONITOR_CONTROL_PRIVATE_H

#define kIPMonitorControlServerName \
    "com.apple.SystemConfiguration.IPMonitorControl"

typedef CF_ENUM(uint32_t, IPMonitorControlRequestType) {
    kIPMonitorControlRequestTypeNone					= 0,
    kIPMonitorControlRequestTypeSetInterfaceRank			= 1,
    kIPMonitorControlRequestTypeGetInterfaceRank			= 2,
    kIPMonitorControlRequestTypeSetInterfaceAdvisory			= 3,
    kIPMonitorControlRequestTypeInterfaceAdvisoryIsSet			= 4,
    kIPMonitorControlRequestTypeAnyInterfaceAdvisoryIsSet	 	= 5,
    kIPMonitorControlRequestTypeGetInterfaceRankAssertionInfo 		= 6,
    kIPMonitorControlRequestTypeGetInterfaceAdvisoryInfo 		= 7,
    kIPMonitorControlRequestTypeGetInterfaceRankAssertionInterfaceNames	= 8,
    kIPMonitorControlRequestTypeGetInterfaceAdvisoryInterfaceNames 	= 9,
};

/*
 * kIPMonitorControlRequestKey*
 * - keys used to communicate a request to the server
 */
#define kIPMonitorControlRequestKeyType			"Type"
#define kIPMonitorControlRequestKeyProcessName		"ProcessName"
#define kIPMonitorControlRequestKeyInterfaceName	"InterfaceName"
#define kIPMonitorControlRequestKeyPrimaryRank		"PrimaryRank"
#define kIPMonitorControlRequestKeyAdvisory		"Advisory"
#define kIPMonitorControlRequestKeyReason		"Reason"

/*
 * kIPMonitorControlResponseKey*
 * - keys used to communicate the response from the server
 */
#define kIPMonitorControlResponseKeyError		"Error"
#define kIPMonitorControlResponseKeyPrimaryRank		"PrimaryRank"
#define kIPMonitorControlResponseKeyAdvisoryIsSet	"AdvisoryIsSet"
#define kIPMonitorControlResponseKeyRankAssertionInfo	"RankAssertionInfo"
#define kIPMonitorControlResponseKeyAdvisoryInfo	"AdvisoryInfo"
#define kIPMonitorControlResponseKeyInterfaceNames	"InterfaceNames"

/*
 * kIPMonitorControlRankAssertionInfoKey*
 * - keys used in the individual rank assertion info dictionaries
 */
#define kIPMonitorControlRankAssertionInfoPrimaryRank	"PrimaryRank"
#define kIPMonitorControlRankAssertionInfoProcessID	"ProcessID"
#define kIPMonitorControlRankAssertionInfoProcessName	"ProcessName"

/*
 * kIPMonitorControlAdvisoryInfoKey*
 * - keys used in the individual advisory info dictionaries
 */
#define kIPMonitorControlAdvisoryInfoAdvisory		"Advisory"
#define kIPMonitorControlAdvisoryInfoProcessID		"ProcessID"
#define kIPMonitorControlAdvisoryInfoProcessName	"ProcessName"

static inline CFStringRef
_IPMonitorControlCopyInterfaceAdvisoryNotificationKey(CFStringRef ifname)
{
    return SCDynamicStoreKeyCreateNetworkInterfaceEntity(NULL,
							 kSCDynamicStoreDomainState,
							 ifname,
							 CFSTR("Advisory"));
}

static inline CFStringRef
_IPMonitorControlCopyInterfaceRankAssertionNotificationKey(CFStringRef ifname)
{
    return SCDynamicStoreKeyCreateNetworkInterfaceEntity(NULL,
							 kSCDynamicStoreDomainState,
							 ifname,
							 CFSTR("RankAssertion"));
}

static inline void
my_CFRelease(void * t)
{
    void * * obj = (void * *)t;
    if (obj && *obj) {
	CFRelease(*obj);
	*obj = NULL;
    }
    return;
}

#endif /* _IPMONITOR_CONTROL_PRIVATE_H */
