/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
#include "WKWebsiteDataStoreConfigurationRef.h"

#include "WKAPICast.h"
#include "WebsiteDataStoreConfiguration.h"

WKTypeID WKWebsiteDataStoreConfigurationGetTypeID()
{
    return WebKit::toAPI(WebKit::WebsiteDataStoreConfiguration::APIType);
}

WKWebsiteDataStoreConfigurationRef WKWebsiteDataStoreConfigurationCreate()
{
#if PLATFORM(COCOA)
    auto configuration = WebKit::WebsiteDataStoreConfiguration::create(WebKit::IsPersistent::Yes);
#else
    auto configuration = WebKit::WebsiteDataStoreConfiguration::createWithBaseDirectories(nullString(), nullString());
#endif
    return toAPI(&configuration.leakRef());
}

WKStringRef WKWebsiteDataStoreConfigurationCopyApplicationCacheDirectory(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->applicationCacheDirectory());
}

void WKWebsiteDataStoreConfigurationSetApplicationCacheDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory)
{
    WebKit::toImpl(configuration)->setApplicationCacheDirectory(WebKit::toImpl(directory)->string());
}

WKStringRef WKWebsiteDataStoreConfigurationCopyNetworkCacheDirectory(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->networkCacheDirectory());
}

void WKWebsiteDataStoreConfigurationSetNetworkCacheDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory)
{
    WebKit::toImpl(configuration)->setNetworkCacheDirectory(WebKit::toImpl(directory)->string());
}

WKStringRef WKWebsiteDataStoreConfigurationCopyIndexedDBDatabaseDirectory(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->indexedDBDatabaseDirectory());
}

void WKWebsiteDataStoreConfigurationSetIndexedDBDatabaseDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory)
{
    WebKit::toImpl(configuration)->setIndexedDBDatabaseDirectory(WebKit::toImpl(directory)->string());
}

WKStringRef WKWebsiteDataStoreConfigurationCopyLocalStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->localStorageDirectory());
}

void WKWebsiteDataStoreConfigurationSetLocalStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory)
{
    WebKit::toImpl(configuration)->setLocalStorageDirectory(WebKit::toImpl(directory)->string());
}

WKStringRef WKWebsiteDataStoreConfigurationCopyWebSQLDatabaseDirectory(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->webSQLDatabaseDirectory());
}

void WKWebsiteDataStoreConfigurationSetWebSQLDatabaseDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory)
{
    WebKit::toImpl(configuration)->setWebSQLDatabaseDirectory(WebKit::toImpl(directory)->string());
}

WKStringRef WKWebsiteDataStoreConfigurationCopyCacheStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->cacheStorageDirectory());
}

void WKWebsiteDataStoreConfigurationSetCacheStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory)
{
    WebKit::toImpl(configuration)->setCacheStorageDirectory(WebKit::toImpl(directory)->string());
}

WKStringRef WKWebsiteDataStoreConfigurationCopyGeneralStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->generalStorageDirectory());
}

void WKWebsiteDataStoreConfigurationSetGeneralStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory)
{
    WebKit::toImpl(configuration)->setGeneralStorageDirectory(WebKit::toImpl(directory)->string());
}

WKStringRef WKWebsiteDataStoreConfigurationCopyMediaKeysStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->mediaKeysStorageDirectory());
}

void WKWebsiteDataStoreConfigurationSetMediaKeysStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory)
{
    WebKit::toImpl(configuration)->setMediaKeysStorageDirectory(WebKit::toImpl(directory)->string());
}

WKStringRef WKWebsiteDataStoreConfigurationCopyResourceLoadStatisticsDirectory(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->resourceLoadStatisticsDirectory());
}

void WKWebsiteDataStoreConfigurationSetResourceLoadStatisticsDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory)
{
    WebKit::toImpl(configuration)->setResourceLoadStatisticsDirectory(WebKit::toImpl(directory)->string());
}

WKStringRef WKWebsiteDataStoreConfigurationCopyServiceWorkerRegistrationDirectory(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->serviceWorkerRegistrationDirectory());
}

void WKWebsiteDataStoreConfigurationSetServiceWorkerRegistrationDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory)
{
    WebKit::toImpl(configuration)->setServiceWorkerRegistrationDirectory(WebKit::toImpl(directory)->string());
}

WKStringRef WKWebsiteDataStoreConfigurationCopyCookieStorageFile(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->cookieStorageFile());
}

void WKWebsiteDataStoreConfigurationSetCookieStorageFile(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef cookieStorageFile)
{
    WebKit::toImpl(configuration)->setCookieStorageFile(WebKit::toImpl(cookieStorageFile)->string());
}

uint64_t WKWebsiteDataStoreConfigurationGetPerOriginStorageQuota(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toImpl(configuration)->perOriginStorageQuota();
}

void WKWebsiteDataStoreConfigurationSetPerOriginStorageQuota(WKWebsiteDataStoreConfigurationRef configuration, uint64_t quota)
{
    WebKit::toImpl(configuration)->setPerOriginStorageQuota(quota);
}

bool WKWebsiteDataStoreConfigurationGetNetworkCacheSpeculativeValidationEnabled(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toImpl(configuration)->networkCacheSpeculativeValidationEnabled();
}

void WKWebsiteDataStoreConfigurationSetNetworkCacheSpeculativeValidationEnabled(WKWebsiteDataStoreConfigurationRef configuration, bool enabled)
{
    WebKit::toImpl(configuration)->setNetworkCacheSpeculativeValidationEnabled(enabled);
}

bool WKWebsiteDataStoreConfigurationGetTestingSessionEnabled(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toImpl(configuration)->testingSessionEnabled();
}

void WKWebsiteDataStoreConfigurationSetTestingSessionEnabled(WKWebsiteDataStoreConfigurationRef configuration, bool enabled)
{
    WebKit::toImpl(configuration)->setTestingSessionEnabled(enabled);
}

bool WKWebsiteDataStoreConfigurationGetStaleWhileRevalidateEnabled(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toImpl(configuration)->staleWhileRevalidateEnabled();
}

void WKWebsiteDataStoreConfigurationSetStaleWhileRevalidateEnabled(WKWebsiteDataStoreConfigurationRef configuration, bool enabled)
{
    WebKit::toImpl(configuration)->setStaleWhileRevalidateEnabled(enabled);
}

WKStringRef WKWebsiteDataStoreConfigurationCopyPCMMachServiceName(WKWebsiteDataStoreConfigurationRef configuration)
{
    return WebKit::toCopiedAPI(WebKit::toImpl(configuration)->pcmMachServiceName());
}

void WKWebsiteDataStoreConfigurationSetPCMMachServiceName(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef name)
{
    WebKit::toImpl(configuration)->setPCMMachServiceName(name ? WebKit::toImpl(name)->string() : String());
}

bool WKWebsiteDataStoreConfigurationHasOriginQuotaRatio(WKWebsiteDataStoreConfigurationRef configuration)
{
    return !!WebKit::toImpl(configuration)->originQuotaRatio();
}

void WKWebsiteDataStoreConfigurationClearOriginQuotaRatio(WKWebsiteDataStoreConfigurationRef configuration)
{
    WebKit::toImpl(configuration)->setOriginQuotaRatio(std::nullopt);
}

bool WKWebsiteDataStoreConfigurationHasTotalQuotaRatio(WKWebsiteDataStoreConfigurationRef configuration)
{
    return !!WebKit::toImpl(configuration)->totalQuotaRatio();
}

void WKWebsiteDataStoreConfigurationClearTotalQuotaRatio(WKWebsiteDataStoreConfigurationRef configuration)
{
    WebKit::toImpl(configuration)->setTotalQuotaRatio(std::nullopt);
}
