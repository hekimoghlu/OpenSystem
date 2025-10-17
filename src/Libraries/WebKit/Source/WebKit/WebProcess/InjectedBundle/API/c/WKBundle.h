/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
#ifndef WKBundle_h
#define WKBundle_h

#include <JavaScriptCore/JavaScript.h>
#include <WebKit/WKBase.h>
#include <WebKit/WKDeprecated.h>

#ifdef __cplusplus
extern "C" {
#endif

// Client
typedef void (*WKBundleDidCreatePageCallback)(WKBundleRef bundle, WKBundlePageRef page, const void* clientInfo);
typedef void (*WKBundleWillDestroyPageCallback)(WKBundleRef bundle, WKBundlePageRef page, const void* clientInfo);
typedef void (*WKBundleDidInitializePageGroupCallback)(WKBundleRef bundle, WKBundlePageGroupRef pageGroup, const void* clientInfo);
typedef void (*WKBundleDidReceiveMessageCallback)(WKBundleRef bundle, WKStringRef name, WKTypeRef messageBody, const void* clientInfo);
typedef void (*WKBundleDidReceiveMessageToPageCallback)(WKBundleRef bundle, WKBundlePageRef page, WKStringRef name, WKTypeRef messageBody, const void* clientInfo);

typedef struct WKBundleClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKBundleClientBase;

typedef struct WKBundleClientV0 {
    WKBundleClientBase                                                  base;

    // Version 0.
    WKBundleDidCreatePageCallback                                       didCreatePage;
    WKBundleWillDestroyPageCallback                                     willDestroyPage;
    WKBundleDidInitializePageGroupCallback                              didInitializePageGroup;
    WKBundleDidReceiveMessageCallback                                   didReceiveMessage;
} WKBundleClientV0;

typedef struct WKBundleClientV1 {
    WKBundleClientBase                                                  base;

    // Version 0.
    WKBundleDidCreatePageCallback                                       didCreatePage;
    WKBundleWillDestroyPageCallback                                     willDestroyPage;
    WKBundleDidInitializePageGroupCallback                              didInitializePageGroup;
    WKBundleDidReceiveMessageCallback                                   didReceiveMessage;

    // Version 1.
    WKBundleDidReceiveMessageToPageCallback                             didReceiveMessageToPage;
} WKBundleClientV1;

WK_EXPORT WKTypeID WKBundleGetTypeID();

WK_EXPORT void WKBundleSetClient(WKBundleRef bundle, WKBundleClientBase* client);
WK_EXPORT void WKBundleSetServiceWorkerProxyCreationCallback(WKBundleRef bundle, void (*)(uint64_t));

WK_EXPORT void WKBundlePostMessage(WKBundleRef bundle, WKStringRef messageName, WKTypeRef messageBody);
WK_EXPORT void WKBundlePostSynchronousMessage(WKBundleRef bundle, WKStringRef messageName, WKTypeRef messageBody, WKTypeRef* returnRetainedData) WK_C_API_DEPRECATED;

WK_EXPORT void WKBundleReportException(JSContextRef, JSValueRef exception);

#ifdef __cplusplus
}
#endif

#endif /* WKBundle_h */
