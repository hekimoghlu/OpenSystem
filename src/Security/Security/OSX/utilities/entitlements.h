/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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
#ifndef _utilities_entitlements_h
#define _utilities_entitlements_h

#include <CoreFoundation/CoreFoundation.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

/// Checks an entitlement dictionary to determine if any Catalyst-related entitlements need to be updated.
bool needsCatalystEntitlementFixup(CFDictionaryRef entitlements);

/// Modifies an entitlements dictionary to add the necessary Catalyst-related entitlements based on pre-existing entitlements.
/// Returns whether the entitlements were modified.
bool updateCatalystEntitlements(CFMutableDictionaryRef entitlements);

/// Hack to address security vulnerability with osinstallersetupd (rdar://137056540).
/// If osinstallersetupd contains the kTCCServiceSystemPolicyAllFiles entitlement, it should be removed.
///
/// Note: Because this function is called in SecCodeCopySigningInformation, which doesn't do
/// validation, it cannot determine with full confidence whether a given app is actually platform or
/// not. This is why the parameter is called `isLikelyPlatform`.
bool needsOSInstallerSetupdEntitlementsFixup(CFStringRef identifier, bool isLikelyPlatform, CFDictionaryRef entitlements);

/// This function removes the kTCCServiceSystemPolicyAllFiles entitlement if it exists.
/// This should only be called if needsOSInstallerSetupdEntitlementsFixup returns true.
bool updateOSInstallerSetupdEntitlements(CFMutableDictionaryRef entitlement);

__END_DECLS

#endif /* _utilities_entitlements_h */
