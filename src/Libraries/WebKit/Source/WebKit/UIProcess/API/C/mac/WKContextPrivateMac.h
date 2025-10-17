/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#ifndef WKContextPrivateMac_h
#define WKContextPrivateMac_h

#include <WebKit/WKBase.h>
#include <WebKit/WKDeprecated.h>
#include <WebKit/WKPluginLoadPolicy.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT bool WKContextIsPlugInUpdateAvailable(WKContextRef context, WKStringRef plugInBundleIdentifier);

WK_EXPORT void WKContextSetPluginLoadClientPolicy(WKContextRef context, WKPluginLoadClientPolicy policy, WKStringRef host, WKStringRef bundleIdentifier, WKStringRef versionString);
WK_EXPORT void WKContextClearPluginClientPolicies(WKContextRef context);

WK_EXPORT WKDictionaryRef WKContextCopyPlugInInfoForBundleIdentifier(WKContextRef context, WKStringRef plugInBundleIdentifier);

typedef void (^WKContextGetInfoForInstalledPlugInsBlock)(WKArrayRef, WKErrorRef);
WK_EXPORT void WKContextGetInfoForInstalledPlugIns(WKContextRef context, WKContextGetInfoForInstalledPlugInsBlock block);

WK_EXPORT void WKContextResetHSTSHosts(WKContextRef context) WK_C_API_DEPRECATED;
WK_EXPORT void WKContextResetHSTSHostsAddedAfterDate(WKContextRef context, double startDateIntervalSince1970) WK_C_API_DEPRECATED;

WK_EXPORT void WKContextRegisterSchemeForCustomProtocol(WKContextRef context, WKStringRef scheme);
WK_EXPORT void WKContextUnregisterSchemeForCustomProtocol(WKContextRef context, WKStringRef scheme);

/* DEPRECATED -  Please use constants from WKPluginInformation instead. */

/* Value type: WKStringRef */
WK_EXPORT WKStringRef WKPlugInInfoPathKey();

/* Value type: WKStringRef */
WK_EXPORT WKStringRef WKPlugInInfoBundleIdentifierKey();

/* Value type: WKStringRef */
WK_EXPORT WKStringRef WKPlugInInfoVersionKey();

/* Value type: WKUInt64Ref */
WK_EXPORT WKStringRef WKPlugInInfoLoadPolicyKey();

/* Value type: WKBooleanRef */
WK_EXPORT WKStringRef WKPlugInInfoUpdatePastLastBlockedVersionIsKnownAvailableKey();

/* Value type: WKBooleanRef */
WK_EXPORT WKStringRef WKPlugInInfoIsSandboxedKey();

WK_EXPORT bool WKContextHandlesSafeBrowsing();

#ifdef __cplusplus
}
#endif

#endif /* WKContextPrivateMac_h */
