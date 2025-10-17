/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
#ifndef WKPageInjectedBundleClient_h
#define WKPageInjectedBundleClient_h

#include <WebKit/WKBase.h>

typedef void (*WKPageDidReceiveMessageFromInjectedBundleCallback)(WKPageRef page, WKStringRef messageName, WKTypeRef messageBody, const void *clientInfo);
typedef void (*WKPageDidReceiveSynchronousMessageFromInjectedBundleCallback)(WKPageRef page, WKStringRef messageName, WKTypeRef messageBody, WKTypeRef* returnData, const void *clientInfo);
typedef WKTypeRef (*WKPageGetInjectedBundleInitializationUserDataCallback)(WKPageRef page, const void *clientInfo);
typedef void (*WKPageDidReceiveSynchronousMessageFromInjectedBundleWithListenerCallback)(WKPageRef page, WKStringRef messageName, WKTypeRef messageBody, WKMessageListenerRef listener, const void* clientInfo);
typedef void (*WKPageDidReceiveAsyncMessageFromInjectedBundleCallback)(WKPageRef page, WKStringRef messageName, WKTypeRef messageBody, WKMessageListenerRef listener, const void* clientInfo);

typedef struct WKPageInjectedBundleClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKPageInjectedBundleClientBase;

typedef struct WKPageInjectedBundleClientV0 {
    WKPageInjectedBundleClientBase                                   base;

    // Version 0.
    WKPageDidReceiveMessageFromInjectedBundleCallback                didReceiveMessageFromInjectedBundle;
    WKPageDidReceiveSynchronousMessageFromInjectedBundleCallback     didReceiveSynchronousMessageFromInjectedBundle;
} WKPageInjectedBundleClientV0;

typedef struct WKPageInjectedBundleClientV1 {
    WKPageInjectedBundleClientBase                                   base;

    // Version 0.
    WKPageDidReceiveMessageFromInjectedBundleCallback                didReceiveMessageFromInjectedBundle;
    WKPageDidReceiveSynchronousMessageFromInjectedBundleCallback     didReceiveSynchronousMessageFromInjectedBundle;

    // Version 1.
    WKPageDidReceiveSynchronousMessageFromInjectedBundleWithListenerCallback didReceiveSynchronousMessageFromInjectedBundleWithListener;
    WKPageDidReceiveAsyncMessageFromInjectedBundleCallback didReceiveAsyncMessageFromInjectedBundle;
} WKPageInjectedBundleClientV1;

#endif // WKPageInjectedBundleClient_h
