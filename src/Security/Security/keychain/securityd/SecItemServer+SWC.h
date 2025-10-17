/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#include <TargetConditionals.h>

#if TARGET_OS_IOS && !TARGET_OS_BRIDGE
#include <CoreFoundation/CoreFoundation.h>

// Compatibility wrappers for SWC Objective-C interface.
typedef enum SecSWCFlags {
	kSecSWCFlags_None = 0,
	kSecSWCFlag_UserApproved = (1 << 0),
	kSecSWCFlag_UserDenied = (1 << 1),
	kSecSWCFlag_SiteApproved = (1 << 2),
	kSecSWCFlag_SiteDenied = (1 << 3),
} SecSWCFlags;

extern SecSWCFlags _SecAppDomainApprovalStatus(CFStringRef appID, CFStringRef fqdn, CFErrorRef *error);
extern void _SecSetAppDomainApprovalStatus(CFStringRef appID, CFStringRef fqdn, CFBooleanRef approved);

extern CFTypeRef _SecCopyFQDNObjectFromString(CFStringRef entitlementValue);
extern CFStringRef _SecGetFQDNFromFQDNObject(CFTypeRef fqdnObject, SInt32 *outPort);
#if !TARGET_OS_SIMULATOR
extern bool _SecEntitlementContainsDomainForService(CFArrayRef domains, CFStringRef domain, SInt32 port);
#endif /* !TARGET_OS_SIMULATOR */
#endif // TARGET_OS_IOS && !TARGET_OS_BRIDGE
