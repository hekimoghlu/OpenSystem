/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#include "APINavigation.h"

#include "WebBackForwardListFrameItem.h"
#include "WebBackForwardListItem.h"
#include <WebCore/RegistrableDomain.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>
#include <wtf/DebugUtilities.h>
#include <wtf/HexNumber.h>
#include <wtf/text/MakeString.h>

namespace API {
using namespace WebCore;
using namespace WebKit;

static constexpr Seconds navigationActivityTimeout { 30_s };

SubstituteData::SubstituteData(Vector<uint8_t>&& content, const ResourceResponse& response, WebCore::SubstituteData::SessionHistoryVisibility sessionHistoryVisibility)
    : SubstituteData(WTFMove(content), response.mimeType(), response.textEncodingName(), response.url().string(), nullptr, sessionHistoryVisibility)
{
}


Navigation::Navigation(WebCore::ProcessIdentifier processID)
    : m_navigationID(WebCore::NavigationIdentifier::generate())
    , m_processID(processID)
    , m_clientNavigationActivity(ProcessThrottler::TimedActivity::create(navigationActivityTimeout))
{
}

Navigation::Navigation(WebCore::ProcessIdentifier processID, RefPtr<WebBackForwardListItem>&& currentAndTargetItem)
    : m_navigationID(WebCore::NavigationIdentifier::generate())
    , m_processID(processID)
    , m_reloadItem(WTFMove(currentAndTargetItem))
    , m_clientNavigationActivity(ProcessThrottler::TimedActivity::create(navigationActivityTimeout))
{
}

Navigation::Navigation(WebCore::ProcessIdentifier processID, WebCore::ResourceRequest&& request, RefPtr<WebBackForwardListItem>&& fromItem)
    : m_navigationID(WebCore::NavigationIdentifier::generate())
    , m_processID(processID)
    , m_originalRequest(WTFMove(request))
    , m_currentRequest(m_originalRequest)
    , m_redirectChain { m_originalRequest.url() }
    , m_fromItem(WTFMove(fromItem))
    , m_clientNavigationActivity(ProcessThrottler::TimedActivity::create(navigationActivityTimeout))
{
}

Navigation::Navigation(WebCore::ProcessIdentifier processID, Ref<WebBackForwardListFrameItem>&& targetFrameItem, RefPtr<WebBackForwardListItem>&& fromItem, FrameLoadType backForwardFrameLoadType)
    : m_navigationID(WebCore::NavigationIdentifier::generate())
    , m_processID(processID)
    , m_originalRequest(targetFrameItem->protectedMainFrame()->url())
    , m_currentRequest(m_originalRequest)
    , m_targetFrameItem(WTFMove(targetFrameItem))
    , m_fromItem(WTFMove(fromItem))
    , m_backForwardFrameLoadType(backForwardFrameLoadType)
    , m_clientNavigationActivity(ProcessThrottler::TimedActivity::create(navigationActivityTimeout))
{
}

Navigation::Navigation(WebCore::ProcessIdentifier processID, std::unique_ptr<SubstituteData>&& substituteData)
    : Navigation(processID)
{
    ASSERT(substituteData);
    m_substituteData = WTFMove(substituteData);
}

Navigation::Navigation(WebCore::ProcessIdentifier processID, WebCore::ResourceRequest&& simulatedRequest, std::unique_ptr<SubstituteData>&& substituteData, RefPtr<WebKit::WebBackForwardListItem>&& fromItem)
    : Navigation(processID, WTFMove(simulatedRequest), WTFMove(fromItem))
{
    ASSERT(substituteData);
    m_substituteData = WTFMove(substituteData);
}

Navigation::~Navigation()
{
}

void Navigation::setCurrentRequest(ResourceRequest&& request, ProcessIdentifier processIdentifier)
{
    m_currentRequest = WTFMove(request);
    m_currentRequestProcessIdentifier = processIdentifier;
}

void Navigation::appendRedirectionURL(const WTF::URL& url)
{
    if (m_redirectChain.isEmpty() || m_redirectChain.last() != url)
        m_redirectChain.append(url);
}

bool Navigation::currentRequestIsCrossSiteRedirect() const
{
    return currentRequestIsRedirect()
        && RegistrableDomain(m_lastNavigationAction.redirectResponse.url()) != RegistrableDomain(m_currentRequest.url());
}

WebKit::WebBackForwardListItem* Navigation::targetItem() const
{
    return m_targetFrameItem ? m_targetFrameItem->backForwardListItem() : nullptr;
}

#if !LOG_DISABLED

WTF::String Navigation::loggingString() const
{
    RefPtr targetItem = this->targetItem();
    return makeString("Most recent URL: "_s, m_currentRequest.url().string(), " Back/forward list item URL: '"_s, targetItem ? targetItem->url() : WTF::String { }, "' (0x"_s, hex(reinterpret_cast<uintptr_t>(targetItem.get())), ')');
}

#endif

} // namespace API
