/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#include "config.h"
#include "WKWebsitePolicies.h"

#include "APIDictionary.h"
#include "APIWebsitePolicies.h"
#include "WKAPICast.h"
#include "WKArray.h"
#include "WKDictionary.h"
#include "WKRetainPtr.h"
#include "WebsiteDataStore.h"
#include <WebCore/DocumentLoader.h>

using namespace WebKit;

WKTypeID WKWebsitePoliciesGetTypeID()
{
    return toAPI(API::WebsitePolicies::APIType);
}

WKWebsitePoliciesRef WKWebsitePoliciesCreate()
{
    return toAPI(&API::WebsitePolicies::create().leakRef());
}

void WKWebsitePoliciesSetContentBlockersEnabled(WKWebsitePoliciesRef websitePolicies, bool enabled)
{
    auto defaultEnablement = enabled ? WebCore::ContentExtensionDefaultEnablement::Enabled : WebCore::ContentExtensionDefaultEnablement::Disabled;
    toImpl(websitePolicies)->setContentExtensionEnablement({ defaultEnablement, { } });
}

bool WKWebsitePoliciesGetContentBlockersEnabled(WKWebsitePoliciesRef websitePolicies)
{
    return toImpl(websitePolicies)->contentExtensionEnablement().first == WebCore::ContentExtensionDefaultEnablement::Enabled;
}

WK_EXPORT WKDictionaryRef WKWebsitePoliciesCopyCustomHeaderFields(WKWebsitePoliciesRef)
{
    return nullptr;
}

WK_EXPORT void WKWebsitePoliciesSetCustomHeaderFields(WKWebsitePoliciesRef, WKDictionaryRef)
{
}

void WKWebsitePoliciesSetAllowedAutoplayQuirks(WKWebsitePoliciesRef websitePolicies, WKWebsiteAutoplayQuirk allowedQuirks)
{
    OptionSet<WebsiteAutoplayQuirk> quirks;
    if (allowedQuirks & kWKWebsiteAutoplayQuirkInheritedUserGestures)
        quirks.add(WebsiteAutoplayQuirk::InheritedUserGestures);

    if (allowedQuirks & kWKWebsiteAutoplayQuirkSynthesizedPauseEvents)
        quirks.add(WebsiteAutoplayQuirk::SynthesizedPauseEvents);

    if (allowedQuirks & kWKWebsiteAutoplayQuirkArbitraryUserGestures)
        quirks.add(WebsiteAutoplayQuirk::ArbitraryUserGestures);

    if (allowedQuirks & kWKWebsiteAutoplayQuirkPerDocumentAutoplayBehavior)
        quirks.add(WebsiteAutoplayQuirk::PerDocumentAutoplayBehavior);

    toImpl(websitePolicies)->setAllowedAutoplayQuirks(quirks);
}

WKWebsiteAutoplayQuirk WKWebsitePoliciesGetAllowedAutoplayQuirks(WKWebsitePoliciesRef websitePolicies)
{
    WKWebsiteAutoplayQuirk quirks = 0;
    auto allowedQuirks = toImpl(websitePolicies)->allowedAutoplayQuirks();

    if (allowedQuirks.contains(WebsiteAutoplayQuirk::SynthesizedPauseEvents))
        quirks |= kWKWebsiteAutoplayQuirkSynthesizedPauseEvents;

    if (allowedQuirks.contains(WebsiteAutoplayQuirk::InheritedUserGestures))
        quirks |= kWKWebsiteAutoplayQuirkInheritedUserGestures;

    if (allowedQuirks.contains(WebsiteAutoplayQuirk::ArbitraryUserGestures))
        quirks |= kWKWebsiteAutoplayQuirkArbitraryUserGestures;

    if (allowedQuirks.contains(WebsiteAutoplayQuirk::PerDocumentAutoplayBehavior))
        quirks |= kWKWebsiteAutoplayQuirkPerDocumentAutoplayBehavior;

    return quirks;
}

WKWebsiteAutoplayPolicy WKWebsitePoliciesGetAutoplayPolicy(WKWebsitePoliciesRef websitePolicies)
{
    switch (toImpl(websitePolicies)->autoplayPolicy()) {
    case WebKit::WebsiteAutoplayPolicy::Default:
        return kWKWebsiteAutoplayPolicyDefault;
    case WebsiteAutoplayPolicy::Allow:
        return kWKWebsiteAutoplayPolicyAllow;
    case WebsiteAutoplayPolicy::AllowWithoutSound:
        return kWKWebsiteAutoplayPolicyAllowWithoutSound;
    case WebsiteAutoplayPolicy::Deny:
        return kWKWebsiteAutoplayPolicyDeny;
    }
    ASSERT_NOT_REACHED();
    return kWKWebsiteAutoplayPolicyDefault;
}

void WKWebsitePoliciesSetAutoplayPolicy(WKWebsitePoliciesRef websitePolicies, WKWebsiteAutoplayPolicy policy)
{
    switch (policy) {
    case kWKWebsiteAutoplayPolicyDefault:
        toImpl(websitePolicies)->setAutoplayPolicy(WebsiteAutoplayPolicy::Default);
        return;
    case kWKWebsiteAutoplayPolicyAllow:
        toImpl(websitePolicies)->setAutoplayPolicy(WebsiteAutoplayPolicy::Allow);
        return;
    case kWKWebsiteAutoplayPolicyAllowWithoutSound:
        toImpl(websitePolicies)->setAutoplayPolicy(WebsiteAutoplayPolicy::AllowWithoutSound);
        return;
    case kWKWebsiteAutoplayPolicyDeny:
        toImpl(websitePolicies)->setAutoplayPolicy(WebsiteAutoplayPolicy::Deny);
        return;
    }
    ASSERT_NOT_REACHED();
}

WKWebsitePopUpPolicy WKWebsitePoliciesGetPopUpPolicy(WKWebsitePoliciesRef websitePolicies)
{
    switch (toImpl(websitePolicies)->popUpPolicy()) {
    case WebsitePopUpPolicy::Default:
        return kWKWebsitePopUpPolicyDefault;
    case WebsitePopUpPolicy::Allow:
        return kWKWebsitePopUpPolicyAllow;
    case WebsitePopUpPolicy::Block:
        return kWKWebsitePopUpPolicyBlock;
    }
    ASSERT_NOT_REACHED();
    return kWKWebsitePopUpPolicyDefault;
}

void WKWebsitePoliciesSetPopUpPolicy(WKWebsitePoliciesRef websitePolicies, WKWebsitePopUpPolicy policy)
{
    switch (policy) {
    case kWKWebsitePopUpPolicyDefault:
        toImpl(websitePolicies)->setPopUpPolicy(WebsitePopUpPolicy::Default);
        return;
    case kWKWebsitePopUpPolicyAllow:
        toImpl(websitePolicies)->setPopUpPolicy(WebsitePopUpPolicy::Allow);
        return;
    case kWKWebsitePopUpPolicyBlock:
        toImpl(websitePolicies)->setPopUpPolicy(WebsitePopUpPolicy::Block);
        return;
    }
    ASSERT_NOT_REACHED();
}

WKWebsiteDataStoreRef WKWebsitePoliciesGetDataStore(WKWebsitePoliciesRef websitePolicies)
{
    return toAPI(toImpl(websitePolicies)->websiteDataStore());
}

void WKWebsitePoliciesSetDataStore(WKWebsitePoliciesRef websitePolicies, WKWebsiteDataStoreRef websiteDataStore)
{
    toImpl(websitePolicies)->setWebsiteDataStore(toImpl(websiteDataStore));
}

