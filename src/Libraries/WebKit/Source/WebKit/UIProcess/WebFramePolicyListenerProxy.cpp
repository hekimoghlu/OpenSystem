/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
#include "WebFramePolicyListenerProxy.h"

#include "APINavigation.h"
#include "APIWebsitePolicies.h"
#include "BrowsingWarning.h"
#include "WebFrameProxy.h"
#include "WebsiteDataStore.h"
#include "WebsitePoliciesData.h"

namespace WebKit {

WebFramePolicyListenerProxy::WebFramePolicyListenerProxy(Reply&& reply, ShouldExpectSafeBrowsingResult expectSafeBrowsingResult, ShouldExpectAppBoundDomainResult expectAppBoundDomainResult, ShouldWaitForInitialLinkDecorationFilteringData shouldWaitForInitialLinkDecorationFilteringData)
    : m_reply(WTFMove(reply))
{
    if (expectSafeBrowsingResult == ShouldExpectSafeBrowsingResult::No)
        didReceiveSafeBrowsingResults({ });
    if (expectAppBoundDomainResult == ShouldExpectAppBoundDomainResult::No)
        didReceiveAppBoundDomainResult({ });
    if (shouldWaitForInitialLinkDecorationFilteringData == ShouldWaitForInitialLinkDecorationFilteringData::No)
        didReceiveInitialLinkDecorationFilteringData();
}

WebFramePolicyListenerProxy::~WebFramePolicyListenerProxy() = default;

void WebFramePolicyListenerProxy::didReceiveAppBoundDomainResult(std::optional<NavigatingToAppBoundDomain> isNavigatingToAppBoundDomain)
{
    ASSERT(RunLoop::isMain());

    if (m_policyResult && m_safeBrowsingWarning && m_doneWaitingForLinkDecorationFilteringData) {
        if (m_reply)
            m_reply(WebCore::PolicyAction::Use, m_policyResult->first.get(), m_policyResult->second, WTFMove(*m_safeBrowsingWarning), isNavigatingToAppBoundDomain, WasNavigationIntercepted::No);
    } else
        m_isNavigatingToAppBoundDomain = isNavigatingToAppBoundDomain;
}

void WebFramePolicyListenerProxy::didReceiveSafeBrowsingResults(RefPtr<BrowsingWarning>&& safeBrowsingWarning)
{
    ASSERT(isMainRunLoop());
    ASSERT(!m_safeBrowsingWarning);
    if (m_policyResult && m_isNavigatingToAppBoundDomain && m_doneWaitingForLinkDecorationFilteringData) {
        if (m_reply)
            m_reply(WebCore::PolicyAction::Use, m_policyResult->first.get(), m_policyResult->second, WTFMove(safeBrowsingWarning), *m_isNavigatingToAppBoundDomain, WasNavigationIntercepted::No);
    } else
        m_safeBrowsingWarning = WTFMove(safeBrowsingWarning);
}

void WebFramePolicyListenerProxy::didReceiveInitialLinkDecorationFilteringData()
{
    ASSERT(RunLoop::isMain());
    ASSERT(!m_doneWaitingForLinkDecorationFilteringData);

    if (m_policyResult && m_isNavigatingToAppBoundDomain && m_safeBrowsingWarning) {
        if (m_reply)
            m_reply(WebCore::PolicyAction::Use, m_policyResult->first.get(), m_policyResult->second, WTFMove(*m_safeBrowsingWarning), *m_isNavigatingToAppBoundDomain, WasNavigationIntercepted::No);
        return;
    }

    m_doneWaitingForLinkDecorationFilteringData = true;
}

void WebFramePolicyListenerProxy::use(API::WebsitePolicies* policies, ProcessSwapRequestedByClient processSwapRequestedByClient)
{
    if (m_safeBrowsingWarning && m_isNavigatingToAppBoundDomain && m_doneWaitingForLinkDecorationFilteringData) {
        if (m_reply)
            m_reply(WebCore::PolicyAction::Use, policies, processSwapRequestedByClient, WTFMove(*m_safeBrowsingWarning), *m_isNavigatingToAppBoundDomain, WasNavigationIntercepted::No);
    } else if (!m_policyResult)
        m_policyResult = {{ policies, processSwapRequestedByClient }};
}

void WebFramePolicyListenerProxy::download()
{
    if (m_reply)
        m_reply(WebCore::PolicyAction::Download, nullptr, ProcessSwapRequestedByClient::No, { }, { }, WasNavigationIntercepted::No);
}

void WebFramePolicyListenerProxy::ignore(WasNavigationIntercepted wasNavigationIntercepted)
{
    if (m_reply)
        m_reply(WebCore::PolicyAction::Ignore, nullptr, ProcessSwapRequestedByClient::No, { }, { }, wasNavigationIntercepted);
}

} // namespace WebKit
