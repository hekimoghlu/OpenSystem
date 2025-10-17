/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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
#ifndef WKCookieManager_h
#define WKCookieManager_h

#include <WebKit/WKBase.h>
#include <WebKit/WKDeprecated.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKHTTPCookieAcceptPolicyAlways = 0,
    kWKHTTPCookieAcceptPolicyNever = 1,
    kWKHTTPCookieAcceptPolicyOnlyFromMainDocumentDomain = 2,
    kWKHTTPCookieAcceptPolicyExclusivelyFromMainDocumentDomain = 3
};
typedef uint32_t WKHTTPCookieAcceptPolicy;

// Cookie Manager Client
typedef void (*WKCookieManagerCookiesDidChangeCallback)(WKCookieManagerRef cookieManager, const void *clientInfo);

typedef struct WKCookieManagerClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKCookieManagerClientBase;

typedef struct WKCookieManagerClientV0 {
    WKCookieManagerClientBase                                           base;

    // Version 0.
    WKCookieManagerCookiesDidChangeCallback                             cookiesDidChange;
} WKCookieManagerClientV0;

WK_EXPORT WKTypeID WKCookieManagerGetTypeID() WK_C_API_DEPRECATED;

WK_EXPORT void WKCookieManagerSetClient(WKCookieManagerRef cookieManager, const WKCookieManagerClientBase* client) WK_C_API_DEPRECATED;

typedef void (*WKCookieManagerGetCookieHostnamesFunction)(WKArrayRef, WKErrorRef, void*);
WK_EXPORT void WKCookieManagerGetHostnamesWithCookies(WKCookieManagerRef cookieManager, void* context, WKCookieManagerGetCookieHostnamesFunction function) WK_C_API_DEPRECATED;

WK_EXPORT void WKCookieManagerDeleteCookiesForHostname(WKCookieManagerRef cookieManager, WKStringRef hostname) WK_C_API_DEPRECATED;
WK_EXPORT void WKCookieManagerDeleteAllCookies(WKCookieManagerRef cookieManager) WK_C_API_DEPRECATED_WITH_REPLACEMENT(WKHTTPCookieStoreDeleteAllCookies);

// The time here is relative to the Unix epoch.
WK_EXPORT void WKCookieManagerDeleteAllCookiesModifiedAfterDate(WKCookieManagerRef cookieManager, double) WK_C_API_DEPRECATED;

typedef void (*WKCookieManagerSetHTTPCookieAcceptPolicyFunction)(WKErrorRef, void*);
WK_EXPORT void WKCookieManagerSetHTTPCookieAcceptPolicy(WKCookieManagerRef cookieManager, WKHTTPCookieAcceptPolicy policy, void* context, WKCookieManagerSetHTTPCookieAcceptPolicyFunction callback) WK_C_API_DEPRECATED_WITH_REPLACEMENT(WKHTTPCookieStoreSetHTTPCookieAcceptPolicy);
typedef void (*WKCookieManagerGetHTTPCookieAcceptPolicyFunction)(WKHTTPCookieAcceptPolicy, WKErrorRef, void*);
WK_EXPORT void WKCookieManagerGetHTTPCookieAcceptPolicy(WKCookieManagerRef cookieManager, void* context, WKCookieManagerGetHTTPCookieAcceptPolicyFunction callback) WK_C_API_DEPRECATED;

WK_EXPORT void WKCookieManagerStartObservingCookieChanges(WKCookieManagerRef cookieManager) WK_C_API_DEPRECATED;
WK_EXPORT void WKCookieManagerStopObservingCookieChanges(WKCookieManagerRef cookieManager) WK_C_API_DEPRECATED;

#ifdef __cplusplus
}
#endif

#endif // WKCookieManager_h
