/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
#pragma once

#include <WebKit/WKBase.h>
#include <WebKit/WKDeprecated.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT WKTypeID WKWebsitePoliciesGetTypeID();

enum WKWebsiteAutoplayPolicy {
    kWKWebsiteAutoplayPolicyDefault,
    kWKWebsiteAutoplayPolicyAllow,
    kWKWebsiteAutoplayPolicyAllowWithoutSound,
    kWKWebsiteAutoplayPolicyDeny
};

typedef uint32_t WKWebsiteAutoplayQuirk;
enum {
    kWKWebsiteAutoplayQuirkSynthesizedPauseEvents = 1 << 0,
    kWKWebsiteAutoplayQuirkInheritedUserGestures = 1 << 1,
    kWKWebsiteAutoplayQuirkArbitraryUserGestures = 1 << 2,
    kWKWebsiteAutoplayQuirkPerDocumentAutoplayBehavior = 1 << 3,
};

enum WKWebsitePopUpPolicy {
    kWKWebsitePopUpPolicyDefault,
    kWKWebsitePopUpPolicyAllow,
    kWKWebsitePopUpPolicyBlock,
};

WK_EXPORT WKWebsitePoliciesRef WKWebsitePoliciesCreate();

WK_EXPORT bool WKWebsitePoliciesGetContentBlockersEnabled(WKWebsitePoliciesRef);
WK_EXPORT void WKWebsitePoliciesSetContentBlockersEnabled(WKWebsitePoliciesRef, bool);

WK_EXPORT WKDictionaryRef WKWebsitePoliciesCopyCustomHeaderFields(WKWebsitePoliciesRef) WK_C_API_DEPRECATED;
WK_EXPORT void WKWebsitePoliciesSetCustomHeaderFields(WKWebsitePoliciesRef, WKDictionaryRef) WK_C_API_DEPRECATED;

WK_EXPORT WKWebsiteAutoplayQuirk WKWebsitePoliciesGetAllowedAutoplayQuirks(WKWebsitePoliciesRef);
WK_EXPORT void WKWebsitePoliciesSetAllowedAutoplayQuirks(WKWebsitePoliciesRef, WKWebsiteAutoplayQuirk);

WK_EXPORT enum WKWebsiteAutoplayPolicy WKWebsitePoliciesGetAutoplayPolicy(WKWebsitePoliciesRef);
WK_EXPORT void WKWebsitePoliciesSetAutoplayPolicy(WKWebsitePoliciesRef, enum WKWebsiteAutoplayPolicy);

WK_EXPORT enum WKWebsitePopUpPolicy WKWebsitePoliciesGetPopUpPolicy(WKWebsitePoliciesRef);
WK_EXPORT void WKWebsitePoliciesSetPopUpPolicy(WKWebsitePoliciesRef, enum WKWebsitePopUpPolicy);

WK_EXPORT WKWebsiteDataStoreRef WKWebsitePoliciesGetDataStore(WKWebsitePoliciesRef);
WK_EXPORT void WKWebsitePoliciesSetDataStore(WKWebsitePoliciesRef, WKWebsiteDataStoreRef);

#ifdef __cplusplus
}
#endif
