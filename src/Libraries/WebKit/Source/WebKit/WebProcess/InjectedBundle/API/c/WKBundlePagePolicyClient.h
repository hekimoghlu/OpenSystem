/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
#ifndef WKBundlePagePolicyClient_h
#define WKBundlePagePolicyClient_h

#include <WebKit/WKBase.h>

enum {
    WKBundlePagePolicyActionPassThrough,
    WKBundlePagePolicyActionUse
};
typedef uint32_t WKBundlePagePolicyAction;

typedef WKBundlePagePolicyAction (*WKBundlePageDecidePolicyForNavigationActionCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKBundleNavigationActionRef navigationAction, WKURLRequestRef request, WKTypeRef* userData, const void* clientInfo);
typedef WKBundlePagePolicyAction (*WKBundlePageDecidePolicyForNewWindowActionCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKBundleNavigationActionRef navigationAction, WKURLRequestRef request, WKStringRef frameName, WKTypeRef* userData, const void* clientInfo);
typedef WKBundlePagePolicyAction (*WKBundlePageDecidePolicyForResponseCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKURLResponseRef response, WKURLRequestRef request, WKTypeRef* userData, const void* clientInfo);
typedef void (*WKBundlePageUnableToImplementPolicyCallback)(WKBundlePageRef page, WKBundleFrameRef frame, WKErrorRef error, WKTypeRef* userData, const void* clientInfo);

typedef struct WKBundlePagePolicyClientBase {
    int                                                                 version;
    const void *                                                        clientInfo;
} WKBundlePagePolicyClientBase;

typedef struct WKBundlePagePolicyClientV0 {
    WKBundlePagePolicyClientBase                                        base;

    // Version 0.
    WKBundlePageDecidePolicyForNavigationActionCallback                 decidePolicyForNavigationAction;
    WKBundlePageDecidePolicyForNewWindowActionCallback                  decidePolicyForNewWindowAction;
    WKBundlePageDecidePolicyForResponseCallback                         decidePolicyForResponse;
    WKBundlePageUnableToImplementPolicyCallback                         unableToImplementPolicy;
} WKBundlePagePolicyClientV0;

#endif // WKBundlePagePolicyClient_h
