/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#ifndef WKContextPrivate_h
#define WKContextPrivate_h

#include <WebKit/WKBase.h>
#include <WebKit/WKContext.h>

#ifdef __cplusplus
extern "C" {
#endif

struct WKContextStatistics {
    unsigned wkViewCount;
    unsigned wkPageCount;
    unsigned wkFrameCount;
};
typedef struct WKContextStatistics WKContextStatistics;

WK_EXPORT void WKContextGetGlobalStatistics(WKContextStatistics* statistics);

WK_EXPORT void WKContextSetAdditionalPluginsDirectory(WKContextRef context, WKStringRef pluginsDirectory);

WK_EXPORT void WKContextRegisterURLSchemeAsEmptyDocument(WKContextRef context, WKStringRef urlScheme);

WK_EXPORT void WKContextSetAlwaysUsesComplexTextCodePath(WKContextRef context, bool alwaysUseComplexTextCodePath);

WK_EXPORT void WKContextSetDisableFontSubpixelAntialiasingForTesting(WKContextRef context, bool disable);

WK_EXPORT void WKContextRegisterURLSchemeAsSecure(WKContextRef context, WKStringRef urlScheme);

WK_EXPORT void WKContextRegisterURLSchemeAsBypassingContentSecurityPolicy(WKContextRef context, WKStringRef urlScheme);

WK_EXPORT void WKContextRegisterURLSchemeAsCachePartitioned(WKContextRef context, WKStringRef urlScheme);

WK_EXPORT void WKContextRegisterURLSchemeAsCanDisplayOnlyIfCanRequest(WKContextRef context, WKStringRef urlScheme);

WK_EXPORT void WKContextSetDomainRelaxationForbiddenForURLScheme(WKContextRef context, WKStringRef urlScheme);

WK_EXPORT void WKContextSetCanHandleHTTPSServerTrustEvaluation(WKContextRef context, bool value) WK_C_API_DEPRECATED;

WK_EXPORT void WKContextSetPrewarmsProcessesAutomatically(WKContextRef context, bool value);

WK_EXPORT void WKContextSetDiskCacheSpeculativeValidationEnabled(WKContextRef context, bool value) WK_C_API_DEPRECATED;

WK_EXPORT void WKContextSetIconDatabasePath(WKContextRef context, WKStringRef iconDatabasePath);

WK_EXPORT void WKContextAllowSpecificHTTPSCertificateForHost(WKContextRef context, WKCertificateInfoRef certificate, WKStringRef host) WK_C_API_DEPRECATED;

// FIXME: This is a workaround for testing purposes only and should be removed once a better
// solution has been found for testing.
WK_EXPORT void WKContextDisableProcessTermination(WKContextRef context);
WK_EXPORT void WKContextEnableProcessTermination(WKContextRef context);

WK_EXPORT void WKContextSetHTTPPipeliningEnabled(WKContextRef context, bool enabled);

WK_EXPORT void WKContextWarmInitialProcess(WKContextRef context);

// FIXME: This function is temporary and useful during the development of the NetworkProcess feature.
// At some point it should be removed.
WK_EXPORT void WKContextSetUsesNetworkProcess(WKContextRef, bool);

WK_EXPORT void WKContextTerminateGPUProcess(WKContextRef);
WK_EXPORT void WKContextTerminateServiceWorkers(WKContextRef);

typedef void (*WKContextInvalidMessageFunction)(WKStringRef messageName);
WK_EXPORT void WKContextSetInvalidMessageFunction(WKContextInvalidMessageFunction invalidMessageFunction);
    
WK_EXPORT void WKContextSetMemoryCacheDisabled(WKContextRef, bool disabled);

WK_EXPORT void WKContextSetFontAllowList(WKContextRef, WKArrayRef);

WK_EXPORT void WKContextPreconnectToServer(WKContextRef context, WKURLRef serverURL) WK_C_API_DEPRECATED;

WK_EXPORT void WKContextAddSupportedPlugin(WKContextRef context, WKStringRef domain, WKStringRef name, WKArrayRef mimeTypes, WKArrayRef extensions);
WK_EXPORT void WKContextClearSupportedPlugins(WKContextRef context);

WK_EXPORT void WKContextClearCurrentModifierStateForTesting(WKContextRef context);

WK_EXPORT void WKContextSetUseSeparateServiceWorkerProcess(WKContextRef context, bool forceServiceWorkerProcess);

WK_EXPORT void WKContextSetPrimaryWebsiteDataStore(WKContextRef context, WKWebsiteDataStoreRef websiteDataStore);

WK_EXPORT WKArrayRef WKContextCopyLocalhostAliases(WKContextRef context);
WK_EXPORT void WKContextSetLocalhostAliases(WKContextRef context, WKArrayRef localhostAliases);

#ifdef __cplusplus
}
#endif

#endif /* WKContextPrivate_h */
