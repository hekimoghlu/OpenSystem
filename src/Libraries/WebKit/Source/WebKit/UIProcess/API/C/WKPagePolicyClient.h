/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 12, 2025.
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
#ifndef WKPagePolicyClient_h
#define WKPagePolicyClient_h

#include <WebKit/WKBase.h>
#include <WebKit/WKEvent.h>
#include <WebKit/WKPageLoadTypes.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*WKPageDecidePolicyForNavigationActionCallback)(WKPageRef page, WKFrameRef frame, WKFrameNavigationType navigationType, WKEventModifiers modifiers, WKEventMouseButton mouseButton, WKFrameRef originatingFrame, WKURLRequestRef request, WKFramePolicyListenerRef listener, WKTypeRef userData, const void* clientInfo);
typedef void (*WKPageDecidePolicyForNewWindowActionCallback)(WKPageRef page, WKFrameRef frame, WKFrameNavigationType navigationType, WKEventModifiers modifiers, WKEventMouseButton mouseButton, WKURLRequestRef request, WKStringRef frameName, WKFramePolicyListenerRef listener, WKTypeRef userData, const void* clientInfo);
typedef void (*WKPageDecidePolicyForResponseCallback_deprecatedForUseWithV0)(WKPageRef page, WKFrameRef frame, WKURLResponseRef response, WKURLRequestRef request, WKFramePolicyListenerRef listener, WKTypeRef userData, const void* clientInfo);
typedef void (*WKPageUnableToImplementPolicyCallback)(WKPageRef page, WKFrameRef frame, WKErrorRef error, WKTypeRef userData, const void* clientInfo);

typedef void (*WKPageDecidePolicyForNavigationActionCallback_deprecatedForUseWithV0)(WKPageRef page, WKFrameRef frame, WKFrameNavigationType navigationType, WKEventModifiers modifiers, WKEventMouseButton mouseButton, WKURLRequestRef request, WKFramePolicyListenerRef listener, WKTypeRef userData, const void* clientInfo);
typedef void (*WKPageDecidePolicyForResponseCallback)(WKPageRef page, WKFrameRef frame, WKURLResponseRef response, WKURLRequestRef request, bool canShowMIMEType, WKFramePolicyListenerRef listener, WKTypeRef userData, const void* clientInfo);

typedef struct WKPagePolicyClientBase {
    int                                                                  version;
    const void *                                                         clientInfo;
} WKPagePolicyClientBase;

typedef struct WKPagePolicyClientV0 {
    WKPagePolicyClientBase                                               base;

    // Version 0.
    WKPageDecidePolicyForNavigationActionCallback_deprecatedForUseWithV0 decidePolicyForNavigationAction_deprecatedForUseWithV0;
    WKPageDecidePolicyForNewWindowActionCallback                         decidePolicyForNewWindowAction;
    WKPageDecidePolicyForResponseCallback_deprecatedForUseWithV0         decidePolicyForResponse_deprecatedForUseWithV0;
    WKPageUnableToImplementPolicyCallback                                unableToImplementPolicy;
} WKPagePolicyClientV0;

typedef struct WKPagePolicyClientV1 {
    WKPagePolicyClientBase                                               base;

    // Version 0.
    WKPageDecidePolicyForNavigationActionCallback_deprecatedForUseWithV0 decidePolicyForNavigationAction_deprecatedForUseWithV0;
    WKPageDecidePolicyForNewWindowActionCallback                         decidePolicyForNewWindowAction;
    WKPageDecidePolicyForResponseCallback_deprecatedForUseWithV0         decidePolicyForResponse_deprecatedForUseWithV0;
    WKPageUnableToImplementPolicyCallback                                unableToImplementPolicy;

    // Version 1.
    WKPageDecidePolicyForNavigationActionCallback                        decidePolicyForNavigationAction;
    WKPageDecidePolicyForResponseCallback                                decidePolicyForResponse;
} WKPagePolicyClientV1;

#ifdef __cplusplus
}
#endif

#endif // WKPagePolicyClient_h
