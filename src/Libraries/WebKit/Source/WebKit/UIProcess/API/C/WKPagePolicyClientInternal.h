/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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
#ifndef WKPagePolicyClientInternal_h
#define WKPagePolicyClientInternal_h

#include "WKPagePolicyClient.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*WKPageDecidePolicyForNavigationActionCallback_internal)(WKPageRef page, WKFrameRef frame, WKFrameNavigationType navigationType, WKEventModifiers modifiers, WKEventMouseButton mouseButton, WKFrameRef originatingFrame, WKURLRequestRef originalRequest, WKURLRequestRef request, WKFramePolicyListenerRef listener, WKTypeRef userData, const void* clientInfo);

typedef struct WKPagePolicyClientInternal {
    WKPagePolicyClientBase                                               base;

    // Version 0.
    WKPageDecidePolicyForNavigationActionCallback_deprecatedForUseWithV0 decidePolicyForNavigationAction_deprecatedForUseWithV0;
    WKPageDecidePolicyForNewWindowActionCallback                         decidePolicyForNewWindowAction;
    WKPageDecidePolicyForResponseCallback_deprecatedForUseWithV0         decidePolicyForResponse_deprecatedForUseWithV0;
    WKPageUnableToImplementPolicyCallback                                unableToImplementPolicy;

    // Version 1.
    WKPageDecidePolicyForNavigationActionCallback                        decidePolicyForNavigationAction_deprecatedForUseWithV1;
    WKPageDecidePolicyForResponseCallback                                decidePolicyForResponse;

    // Internal.
    WKPageDecidePolicyForNavigationActionCallback_internal               decidePolicyForNavigationAction;
} WKPagePolicyClientInternal;

#ifdef __cplusplus
}
#endif

#endif // WKPagePolicyClientInternal_h
