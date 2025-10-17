/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#ifndef WKContextInjectedBundleClient_h
#define WKContextInjectedBundleClient_h

#include <WebKit/WKBase.h>

// Injected Bundle Client
typedef void (*WKContextDidReceiveMessageFromInjectedBundleCallback)(WKContextRef context, WKStringRef messageName, WKTypeRef messageBody, const void *clientInfo);
typedef void (*WKContextDidReceiveSynchronousMessageFromInjectedBundleCallback)(WKContextRef context, WKStringRef messageName, WKTypeRef messageBody, WKTypeRef* returnData, const void *clientInfo);
typedef WKTypeRef (*WKContextGetInjectedBundleInitializationUserDataCallback)(WKContextRef context, const void *clientInfo);
typedef void (*WKContextDidReceiveSynchronousMessageFromInjectedBundleWithListenerCallback)(WKContextRef context, WKStringRef messageName, WKTypeRef messageBody, WKMessageListenerRef listener, const void *clientInfo);

typedef struct WKContextInjectedBundleClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKContextInjectedBundleClientBase;

typedef struct WKContextInjectedBundleClientV0 {
    WKContextInjectedBundleClientBase                                   base;

    // Version 0.
    WKContextDidReceiveMessageFromInjectedBundleCallback                didReceiveMessageFromInjectedBundle;
    WKContextDidReceiveSynchronousMessageFromInjectedBundleCallback     didReceiveSynchronousMessageFromInjectedBundle;
} WKContextInjectedBundleClientV0;

typedef struct WKContextInjectedBundleClientV1 {
    WKContextInjectedBundleClientBase                                   base;

    // Version 0.
    WKContextDidReceiveMessageFromInjectedBundleCallback                didReceiveMessageFromInjectedBundle;
    WKContextDidReceiveSynchronousMessageFromInjectedBundleCallback     didReceiveSynchronousMessageFromInjectedBundle;

    // Version 1.
    WKContextGetInjectedBundleInitializationUserDataCallback            getInjectedBundleInitializationUserData;
} WKContextInjectedBundleClientV1;

typedef struct WKContextInjectedBundleClientV2 {
    WKContextInjectedBundleClientBase                                   base;

    // Version 0.
    WKContextDidReceiveMessageFromInjectedBundleCallback                didReceiveMessageFromInjectedBundle;
    WKContextDidReceiveSynchronousMessageFromInjectedBundleCallback     didReceiveSynchronousMessageFromInjectedBundle;

    // Version 1.
    WKContextGetInjectedBundleInitializationUserDataCallback            getInjectedBundleInitializationUserData;

    // Version 2.
    WKContextDidReceiveSynchronousMessageFromInjectedBundleWithListenerCallback didReceiveSynchronousMessageFromInjectedBundleWithListener;
} WKContextInjectedBundleClientV2;

#endif // WKContextInjectedBundleClient_h
