/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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

#include "config.h"
#include "WebPagePreferencesLockdownModeObserver.h"

#include "APIWebsitePolicies.h"
#include "WKWebpagePreferencesInternal.h"
#include "WebProcessPool.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPagePreferencesLockdownModeObserver);

WebPagePreferencesLockdownModeObserver::WebPagePreferencesLockdownModeObserver(API::WebsitePolicies& policies)
    : m_policies(policies)
{
    addLockdownModeObserver(*this);
}

WebPagePreferencesLockdownModeObserver::~WebPagePreferencesLockdownModeObserver()
{
    removeLockdownModeObserver(*this);
}

RefPtr<API::WebsitePolicies> WebPagePreferencesLockdownModeObserver::protectedPolicies()
{
    return m_policies.get();
}

void WebPagePreferencesLockdownModeObserver::willChangeLockdownMode()
{
    if (auto preferences = wrapper(protectedPolicies().get())) {
        [preferences willChangeValueForKey:@"_captivePortalModeEnabled"];
        [preferences willChangeValueForKey:@"lockdownModeEnabled"];
    }
}

void WebPagePreferencesLockdownModeObserver::didChangeLockdownMode()
{
    if (auto preferences = wrapper(protectedPolicies().get())) {
        [preferences didChangeValueForKey:@"_captivePortalModeEnabled"];
        [preferences didChangeValueForKey:@"lockdownModeEnabled"];
    }
}

void WebPagePreferencesLockdownModeObserver::ref() const
{
    m_policies->ref();
}

void WebPagePreferencesLockdownModeObserver::deref() const
{
    m_policies->deref();
}

} // namespace WebKit
