/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
#ifndef WKWebsiteDataStoreConfigurationRef_h
#define WKWebsiteDataStoreConfigurationRef_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKWebsiteDataStoreConfigurationGetTypeID();

WK_EXPORT WKWebsiteDataStoreConfigurationRef WKWebsiteDataStoreConfigurationCreate();

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyApplicationCacheDirectory(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetApplicationCacheDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory);

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyNetworkCacheDirectory(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetNetworkCacheDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory);

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyIndexedDBDatabaseDirectory(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetIndexedDBDatabaseDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory);

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyLocalStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetLocalStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory);

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyWebSQLDatabaseDirectory(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetWebSQLDatabaseDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory);

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyCacheStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetCacheStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory);

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyGeneralStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetGeneralStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory);

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyMediaKeysStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetMediaKeysStorageDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory);

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyResourceLoadStatisticsDirectory(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetResourceLoadStatisticsDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory);

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyServiceWorkerRegistrationDirectory(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetServiceWorkerRegistrationDirectory(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef directory);

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyCookieStorageFile(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetCookieStorageFile(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef cookieStorageFile);

WK_EXPORT uint64_t WKWebsiteDataStoreConfigurationGetPerOriginStorageQuota(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetPerOriginStorageQuota(WKWebsiteDataStoreConfigurationRef configuration, uint64_t quota);

WK_EXPORT bool WKWebsiteDataStoreConfigurationGetNetworkCacheSpeculativeValidationEnabled(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetNetworkCacheSpeculativeValidationEnabled(WKWebsiteDataStoreConfigurationRef configuration, bool enabled);

WK_EXPORT bool WKWebsiteDataStoreConfigurationGetTestingSessionEnabled(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetTestingSessionEnabled(WKWebsiteDataStoreConfigurationRef configuration, bool enabled);

WK_EXPORT bool WKWebsiteDataStoreConfigurationGetStaleWhileRevalidateEnabled(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetStaleWhileRevalidateEnabled(WKWebsiteDataStoreConfigurationRef configuration, bool enabled);

WK_EXPORT WKStringRef WKWebsiteDataStoreConfigurationCopyPCMMachServiceName(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationSetPCMMachServiceName(WKWebsiteDataStoreConfigurationRef configuration, WKStringRef name);

WK_EXPORT bool WKWebsiteDataStoreConfigurationHasOriginQuotaRatio(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationClearOriginQuotaRatio(WKWebsiteDataStoreConfigurationRef configuration);

WK_EXPORT bool WKWebsiteDataStoreConfigurationHasTotalQuotaRatio(WKWebsiteDataStoreConfigurationRef configuration);
WK_EXPORT void WKWebsiteDataStoreConfigurationClearTotalQuotaRatio(WKWebsiteDataStoreConfigurationRef configuration);

#ifdef __cplusplus
}
#endif

#endif /* WKWebsiteDataStoreConfigurationRef_h */
