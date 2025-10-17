/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 30, 2024.
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
#import "config.h"
#import "WKContextPrivateMac.h"

#import "APIArray.h"
#import "APIDictionary.h"
#import "APINumber.h"
#import "APIString.h"
#import "WKAPICast.h"
#import "WKPluginInformation.h"
#import "WKSharedAPICast.h"
#import "WKStringCF.h"
#import "WebProcessPool.h"
#import <wtf/BlockPtr.h>
#import <wtf/RetainPtr.h>

bool WKContextIsPlugInUpdateAvailable(WKContextRef, WKStringRef)
{
    return false;
}

void WKContextSetPluginLoadClientPolicy(WKContextRef, WKPluginLoadClientPolicy, WKStringRef, WKStringRef, WKStringRef)
{
}

void WKContextClearPluginClientPolicies(WKContextRef)
{
}

WKDictionaryRef WKContextCopyPlugInInfoForBundleIdentifier(WKContextRef contextRef, WKStringRef plugInBundleIdentifierRef)
{
    return 0;
}

void WKContextGetInfoForInstalledPlugIns(WKContextRef contextRef, WKContextGetInfoForInstalledPlugInsBlock block)
{
}

void WKContextResetHSTSHosts(WKContextRef)
{
}

void WKContextResetHSTSHostsAddedAfterDate(WKContextRef, double)
{
}

void WKContextRegisterSchemeForCustomProtocol(WKContextRef context, WKStringRef scheme)
{
    WebKit::WebProcessPool::registerGlobalURLSchemeAsHavingCustomProtocolHandlers(WebKit::toWTFString(scheme));
}

void WKContextUnregisterSchemeForCustomProtocol(WKContextRef context, WKStringRef scheme)
{
    WebKit::WebProcessPool::unregisterGlobalURLSchemeAsHavingCustomProtocolHandlers(WebKit::toWTFString(scheme));
}

/* DEPRECATED -  Please use constants from WKPluginInformation instead. */

WKStringRef WKPlugInInfoPathKey()
{
    return WKPluginInformationPathKey();
}

WKStringRef WKPlugInInfoBundleIdentifierKey()
{
    return WKPluginInformationBundleIdentifierKey();
}

WKStringRef WKPlugInInfoVersionKey()
{
    return WKPluginInformationBundleVersionKey();
}

WKStringRef WKPlugInInfoLoadPolicyKey()
{
    return WKPluginInformationDefaultLoadPolicyKey();
}

WKStringRef WKPlugInInfoUpdatePastLastBlockedVersionIsKnownAvailableKey()
{
    return WKPluginInformationUpdatePastLastBlockedVersionIsKnownAvailableKey();
}

WKStringRef WKPlugInInfoIsSandboxedKey()
{
    return WKPluginInformationHasSandboxProfileKey();
}

bool WKContextHandlesSafeBrowsing()
{
    return true;
}
