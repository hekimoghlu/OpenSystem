/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#include "WebNavigationState.h"

#include "APINavigation.h"
#include "WebPageProxy.h"
#include <WebCore/ResourceRequest.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebNavigationState);

WebNavigationState::WebNavigationState(WebPageProxy& page)
    : m_page(page)
{
}

WebNavigationState::~WebNavigationState()
{
}

Ref<API::Navigation> WebNavigationState::createLoadRequestNavigation(WebCore::ProcessIdentifier processID, ResourceRequest&& request, RefPtr<WebBackForwardListItem>&& currentItem)
{
    auto navigation = API::Navigation::create(processID, WTFMove(request), WTFMove(currentItem));

    m_navigations.set(navigation->navigationID(), navigation.ptr());

    return navigation;
}

Ref<API::Navigation> WebNavigationState::createBackForwardNavigation(WebCore::ProcessIdentifier processID, Ref<WebBackForwardListFrameItem>&& targetFrameItem, RefPtr<WebBackForwardListItem>&& currentItem, FrameLoadType frameLoadType)
{
    Ref navigation = API::Navigation::create(processID, WTFMove(targetFrameItem), WTFMove(currentItem), frameLoadType);

    m_navigations.set(navigation->navigationID(), navigation.ptr());

    return navigation;
}

Ref<API::Navigation> WebNavigationState::createReloadNavigation(WebCore::ProcessIdentifier processID, RefPtr<WebBackForwardListItem>&& currentAndTargetItem)
{
    auto navigation = API::Navigation::create(processID, WTFMove(currentAndTargetItem));

    m_navigations.set(navigation->navigationID(), navigation.ptr());

    return navigation;
}

Ref<API::Navigation> WebNavigationState::createLoadDataNavigation(WebCore::ProcessIdentifier processID, std::unique_ptr<API::SubstituteData>&& substituteData)
{
    auto navigation = API::Navigation::create(processID, WTFMove(substituteData));

    m_navigations.set(navigation->navigationID(), navigation.ptr());

    return navigation;
}

Ref<API::Navigation> WebNavigationState::createSimulatedLoadWithDataNavigation(WebCore::ProcessIdentifier processID, WebCore::ResourceRequest&& request, std::unique_ptr<API::SubstituteData>&& substituteData, RefPtr<WebBackForwardListItem>&& currentItem)
{
    auto navigation = API::Navigation::create(processID, WTFMove(request), WTFMove(substituteData), WTFMove(currentItem));

    m_navigations.set(navigation->navigationID(), navigation.ptr());

    return navigation;
}

API::Navigation* WebNavigationState::navigation(WebCore::NavigationIdentifier navigationID)
{
    return m_navigations.get(navigationID);
}

RefPtr<API::Navigation> WebNavigationState::takeNavigation(WebCore::NavigationIdentifier navigationID)
{
    ASSERT(m_navigations.contains(navigationID));
    
    return m_navigations.take(navigationID);
}

void WebNavigationState::didDestroyNavigation(WebCore::ProcessIdentifier processID, WebCore::NavigationIdentifier navigationID)
{
    auto it = m_navigations.find(navigationID);
    if (it != m_navigations.end() && (*it).value->processID() == processID)
        m_navigations.remove(it);
}

void WebNavigationState::clearAllNavigations()
{
    m_navigations.clear();
}

void WebNavigationState::clearNavigationsFromProcess(WebCore::ProcessIdentifier processID)
{
    Vector<WebCore::NavigationIdentifier> navigationIDsToRemove;
    for (auto& navigation : m_navigations.values()) {
        if (navigation->processID() == processID)
            navigationIDsToRemove.append(navigation->navigationID());
    }
    for (auto navigationID : navigationIDsToRemove)
        m_navigations.remove(navigationID);
}

void WebNavigationState::ref() const
{
    m_page->ref();
}

void WebNavigationState::deref() const
{
    m_page->deref();
}

} // namespace WebKit
