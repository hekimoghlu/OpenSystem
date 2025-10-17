/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#ifndef WKPageConfigurationRef_h
#define WKPageConfigurationRef_h

#include <WebKit/WKBase.h>
#include <WebKit/WKDeprecated.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKPageConfigurationGetTypeID(void);

WK_EXPORT WKPageConfigurationRef WKPageConfigurationCreate(void);

WK_EXPORT WKContextRef WKPageConfigurationGetContext(WKPageConfigurationRef configuration);
WK_EXPORT void WKPageConfigurationSetContext(WKPageConfigurationRef configuration, WKContextRef context);

WK_EXPORT WKPageGroupRef WKPageConfigurationGetPageGroup(WKPageConfigurationRef configuration) WK_C_API_DEPRECATED;
WK_EXPORT void WKPageConfigurationSetPageGroup(WKPageConfigurationRef configuration, WKPageGroupRef pageGroup) WK_C_API_DEPRECATED;

WK_EXPORT WKUserContentControllerRef WKPageConfigurationGetUserContentController(WKPageConfigurationRef configuration);
WK_EXPORT void WKPageConfigurationSetUserContentController(WKPageConfigurationRef configuration, WKUserContentControllerRef userContentController);

WK_EXPORT WKPreferencesRef WKPageConfigurationGetPreferences(WKPageConfigurationRef configuration);
WK_EXPORT void WKPageConfigurationSetPreferences(WKPageConfigurationRef configuration, WKPreferencesRef preferences);

WK_EXPORT WKPageRef WKPageConfigurationGetRelatedPage(WKPageConfigurationRef configuration) WK_C_API_DEPRECATED;
WK_EXPORT void WKPageConfigurationSetRelatedPage(WKPageConfigurationRef configuration, WKPageRef relatedPage) WK_C_API_DEPRECATED;

WK_EXPORT WKWebsiteDataStoreRef WKPageConfigurationGetWebsiteDataStore(WKPageConfigurationRef configuration);
WK_EXPORT void WKPageConfigurationSetWebsiteDataStore(WKPageConfigurationRef configuration, WKWebsiteDataStoreRef websiteDataStore);

WK_EXPORT void WKPageConfigurationSetInitialCapitalizationEnabled(WKPageConfigurationRef configuration, bool enabled);
WK_EXPORT void WKPageConfigurationSetBackgroundCPULimit(WKPageConfigurationRef configuration, double cpuLimit); // Terminates process if it uses more than CPU limit over an extended period of time while in the background.

WK_EXPORT void WKPageConfigurationSetAllowTestOnlyIPC(WKPageConfigurationRef configuration, bool allowTestOnlyIPC);

WK_EXPORT void WKPageConfigurationSetPortsForUpgradingInsecureSchemeForTesting(WKPageConfigurationRef configuration, uint16_t upgradeFromInsecurePort, uint16_t upgradeToSecurePort);

#ifdef __cplusplus
}
#endif

#endif // WKPageConfigurationRef_h
