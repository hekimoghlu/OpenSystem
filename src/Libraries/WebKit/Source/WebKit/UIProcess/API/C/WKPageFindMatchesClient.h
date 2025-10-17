/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#ifndef WKPageFindMatchesClient_h
#define WKPageFindMatchesClient_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKMoreThanMaximumMatchCount = -1
};

// Find match client.
typedef void (*WKPageDidFindStringMatchesCallback)(WKPageRef page, WKStringRef string, WKArrayRef matches, int firstIndex, const void* clientInfo);
typedef void (*WKPageDidGetImageForMatchResultCallback)(WKPageRef page, WKImageRef image, uint32_t index, const void* clientInfo);

typedef struct WKPageFindMatchesClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKPageFindMatchesClientBase;

typedef struct WKPageFindMatchesClientV0 {
    WKPageFindMatchesClientBase                                         base;

    // Version 0.
    WKPageDidFindStringMatchesCallback                                  didFindStringMatches;
    WKPageDidGetImageForMatchResultCallback                             didGetImageForMatchResult;
} WKPageFindMatchesClientV0;

#ifdef __cplusplus
}
#endif


#endif // WKPageFindMatchesClient_h
