/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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
#include "APIWebsitePolicies.h"

#include "WebProcessPool.h"
#include "WebUserContentControllerProxy.h"
#include "WebsiteDataStore.h"
#include "WebsitePoliciesData.h"

#if PLATFORM(COCOA)
#include "WebPagePreferencesLockdownModeObserver.h"
#endif

namespace API {

WebsitePolicies::WebsitePolicies()
#if PLATFORM(COCOA)
    : m_lockdownModeObserver(makeUniqueWithoutRefCountedCheck<WebKit::WebPagePreferencesLockdownModeObserver>(*this))
#endif
{
}

Ref<WebsitePolicies> WebsitePolicies::copy() const
{
    auto policies = WebsitePolicies::create();
    policies->m_data = m_data;
    policies->setWebsiteDataStore(m_websiteDataStore.get());
    policies->setUserContentController(m_userContentController.get());
    policies->setLockdownModeEnabled(m_lockdownModeEnabled);
#if PLATFORM(COCOA)
    policies->m_lockdownModeObserver = makeUniqueWithoutRefCountedCheck<WebKit::WebPagePreferencesLockdownModeObserver>(policies);
#endif
    return policies;
}

WebsitePolicies::~WebsitePolicies() = default;

void WebsitePolicies::setWebsiteDataStore(RefPtr<WebKit::WebsiteDataStore>&& websiteDataStore)
{
    m_websiteDataStore = WTFMove(websiteDataStore);
}

void WebsitePolicies::setUserContentController(RefPtr<WebKit::WebUserContentControllerProxy>&& controller)
{
    m_userContentController = WTFMove(controller);
}

WebKit::WebsitePoliciesData WebsitePolicies::data()
{
    return m_data;
}

bool WebsitePolicies::lockdownModeEnabled() const
{
    return m_lockdownModeEnabled ? *m_lockdownModeEnabled : WebKit::lockdownModeEnabledBySystem();
}

}

