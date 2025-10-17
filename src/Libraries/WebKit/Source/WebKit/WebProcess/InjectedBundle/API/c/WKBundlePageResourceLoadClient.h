/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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
#ifndef WKBundlePageResourceLoadClient_h
#define WKBundlePageResourceLoadClient_h

#include <WebKit/WKBase.h>

typedef void (*WKBundlePageDidInitiateLoadForResourceCallback)(WKBundlePageRef, WKBundleFrameRef, uint64_t resourceIdentifier, WKURLRequestRef, bool pageIsProvisionallyLoading, const void* clientInfo);
typedef WKURLRequestRef (*WKBundlePageWillSendRequestForFrameCallback)(WKBundlePageRef, WKBundleFrameRef, uint64_t resourceIdentifier, WKURLRequestRef, WKURLResponseRef redirectResponse, const void *clientInfo);
typedef void (*WKBundlePageDidReceiveResponseForResourceCallback)(WKBundlePageRef, WKBundleFrameRef, uint64_t resourceIdentifier, WKURLResponseRef, const void* clientInfo);
typedef void (*WKBundlePageDidReceiveContentLengthForResourceCallback)(WKBundlePageRef, WKBundleFrameRef, uint64_t resourceIdentifier, uint64_t contentLength, const void* clientInfo);
typedef void (*WKBundlePageDidFinishLoadForResourceCallback)(WKBundlePageRef, WKBundleFrameRef, uint64_t resourceIdentifier, const void* clientInfo);
typedef void (*WKBundlePageDidFailLoadForResourceCallback)(WKBundlePageRef, WKBundleFrameRef, uint64_t resourceIdentifier, WKErrorRef, const void* clientInfo);
typedef bool (*WKBundlePageShouldCacheResponseCallback)(WKBundlePageRef, WKBundleFrameRef, uint64_t resourceIdentifier, const void* clientInfo);
typedef bool (*WKBundlePageShouldUseCredentialStorageCallback)(WKBundlePageRef, WKBundleFrameRef, uint64_t resourceIdentifier, const void* clientInfo);

typedef struct WKBundlePageResourceLoadClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKBundlePageResourceLoadClientBase;

typedef struct WKBundlePageResourceLoadClientV0 {
    WKBundlePageResourceLoadClientBase                                  base;

    // Version 0.
    WKBundlePageDidInitiateLoadForResourceCallback                      didInitiateLoadForResource;

    // willSendRequestForFrame is supposed to return a retained reference to the URL request.
    WKBundlePageWillSendRequestForFrameCallback                         willSendRequestForFrame;

    WKBundlePageDidReceiveResponseForResourceCallback                   didReceiveResponseForResource;
    WKBundlePageDidReceiveContentLengthForResourceCallback              didReceiveContentLengthForResource;
    WKBundlePageDidFinishLoadForResourceCallback                        didFinishLoadForResource;
    WKBundlePageDidFailLoadForResourceCallback                          didFailLoadForResource;
} WKBundlePageResourceLoadClientV0;

typedef struct WKBundlePageResourceLoadClientV1 {
    WKBundlePageResourceLoadClientBase                                  base;

    // Version 0.
    WKBundlePageDidInitiateLoadForResourceCallback                      didInitiateLoadForResource;

    // willSendRequestForFrame is supposed to return a retained reference to the URL request.
    WKBundlePageWillSendRequestForFrameCallback                         willSendRequestForFrame;

    WKBundlePageDidReceiveResponseForResourceCallback                   didReceiveResponseForResource;
    WKBundlePageDidReceiveContentLengthForResourceCallback              didReceiveContentLengthForResource;
    WKBundlePageDidFinishLoadForResourceCallback                        didFinishLoadForResource;
    WKBundlePageDidFailLoadForResourceCallback                          didFailLoadForResource;

    // Version 1.
    WKBundlePageShouldCacheResponseCallback                             shouldCacheResponse;
    WKBundlePageShouldUseCredentialStorageCallback                      shouldUseCredentialStorage;
} WKBundlePageResourceLoadClientV1;

#endif // WKBundlePageResourceLoadClient_h
