/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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
#include "config.h"
#include "WKPageConfigurationRef.h"

#include "APIPageConfiguration.h"
#include "BrowsingContextGroup.h"
#include "WKAPICast.h"
#include "WebPageGroup.h"
#include "WebPageProxy.h"
#include "WebProcessPool.h"
#include "WebUserContentControllerProxy.h"

using namespace WebKit;

WKTypeID WKPageConfigurationGetTypeID()
{
    return toAPI(API::PageConfiguration::APIType);
}

WKPageConfigurationRef WKPageConfigurationCreate()
{
    return toAPI(&API::PageConfiguration::create().leakRef());
}

WKContextRef WKPageConfigurationGetContext(WKPageConfigurationRef configuration)
{
    return toAPI(toImpl(configuration)->processPool());
}

void WKPageConfigurationSetContext(WKPageConfigurationRef configuration, WKContextRef context)
{
    toImpl(configuration)->setProcessPool(toImpl(context));
}

WKPageGroupRef WKPageConfigurationGetPageGroup(WKPageConfigurationRef)
{
    return nullptr;
}

void WKPageConfigurationSetPageGroup(WKPageConfigurationRef, WKPageGroupRef)
{
}

WKUserContentControllerRef WKPageConfigurationGetUserContentController(WKPageConfigurationRef configuration)
{
    return toAPI(toImpl(configuration)->userContentController());
}

void WKPageConfigurationSetUserContentController(WKPageConfigurationRef configuration, WKUserContentControllerRef userContentController)
{
    toImpl(configuration)->setUserContentController(toImpl(userContentController));
}

WKPreferencesRef WKPageConfigurationGetPreferences(WKPageConfigurationRef configuration)
{
    return toAPI(toImpl(configuration)->preferences());
}

void WKPageConfigurationSetPreferences(WKPageConfigurationRef configuration, WKPreferencesRef preferences)
{
    toImpl(configuration)->setPreferences(toImpl(preferences));
}

WKPageRef WKPageConfigurationGetRelatedPage(WKPageConfigurationRef configuration)
{
    return toAPI(toImpl(configuration)->relatedPage());
}

void WKPageConfigurationSetRelatedPage(WKPageConfigurationRef configuration, WKPageRef relatedPage)
{
    toImpl(configuration)->setRelatedPage(toImpl(relatedPage));
}

WKWebsiteDataStoreRef WKPageConfigurationGetWebsiteDataStore(WKPageConfigurationRef configuration)
{
    return toAPI(toImpl(configuration)->websiteDataStore());
}

void WKPageConfigurationSetWebsiteDataStore(WKPageConfigurationRef configuration, WKWebsiteDataStoreRef websiteDataStore)
{
    toImpl(configuration)->setWebsiteDataStore(toImpl(websiteDataStore));
}

void WKPageConfigurationSetInitialCapitalizationEnabled(WKPageConfigurationRef configuration, bool enabled)
{
    toImpl(configuration)->setInitialCapitalizationEnabled(enabled);
}

void WKPageConfigurationSetBackgroundCPULimit(WKPageConfigurationRef configuration, double cpuLimit)
{
    toImpl(configuration)->setCPULimit(cpuLimit);
}

void WKPageConfigurationSetAllowTestOnlyIPC(WKPageConfigurationRef configuration, bool allowTestOnlyIPC)
{
    toImpl(configuration)->setAllowTestOnlyIPC(allowTestOnlyIPC);
}

void WKPageConfigurationSetPortsForUpgradingInsecureSchemeForTesting(WKPageConfigurationRef configuration, uint16_t upgradeFromInsecurePort, uint16_t upgradeToSecurePort)
{
    toImpl(configuration)->setPortsForUpgradingInsecureSchemeForTesting(upgradeFromInsecurePort, upgradeToSecurePort);
}
