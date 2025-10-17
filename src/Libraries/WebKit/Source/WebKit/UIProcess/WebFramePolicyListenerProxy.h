/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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

#include "APIObject.h"
#include "PolicyDecision.h"
#include <WebCore/FrameLoaderTypes.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Vector.h>

namespace API {
class WebsitePolicies;
}

namespace WebKit {

class BrowsingWarning;

enum class ProcessSwapRequestedByClient : bool { No, Yes };
enum class ShouldExpectSafeBrowsingResult : bool { No, Yes };
enum class ShouldExpectAppBoundDomainResult : bool { No, Yes };
enum class ShouldWaitForInitialLinkDecorationFilteringData : bool { No, Yes };
enum class WasNavigationIntercepted : bool { No, Yes };

class WebFramePolicyListenerProxy : public API::ObjectImpl<API::Object::Type::FramePolicyListener> {
public:

    using Reply = CompletionHandler<void(WebCore::PolicyAction, API::WebsitePolicies*, ProcessSwapRequestedByClient, RefPtr<BrowsingWarning>&&, std::optional<NavigatingToAppBoundDomain>, WasNavigationIntercepted)>;
    static Ref<WebFramePolicyListenerProxy> create(Reply&& reply, ShouldExpectSafeBrowsingResult expectSafeBrowsingResult, ShouldExpectAppBoundDomainResult expectAppBoundDomainResult, ShouldWaitForInitialLinkDecorationFilteringData shouldWaitForInitialLinkDecorationFilteringData)
    {
        return adoptRef(*new WebFramePolicyListenerProxy(WTFMove(reply), expectSafeBrowsingResult, expectAppBoundDomainResult, shouldWaitForInitialLinkDecorationFilteringData));
    }
    ~WebFramePolicyListenerProxy();

    void use(API::WebsitePolicies* = nullptr, ProcessSwapRequestedByClient = ProcessSwapRequestedByClient::No);
    void download();
    void ignore(WasNavigationIntercepted = WasNavigationIntercepted::No);
    
    void didReceiveSafeBrowsingResults(RefPtr<BrowsingWarning>&&);
    void didReceiveAppBoundDomainResult(std::optional<NavigatingToAppBoundDomain>);
    void didReceiveManagedDomainResult(std::optional<NavigatingToAppBoundDomain>);
    void didReceiveInitialLinkDecorationFilteringData();

private:
    WebFramePolicyListenerProxy(Reply&&, ShouldExpectSafeBrowsingResult, ShouldExpectAppBoundDomainResult, ShouldWaitForInitialLinkDecorationFilteringData);

    std::optional<std::pair<RefPtr<API::WebsitePolicies>, ProcessSwapRequestedByClient>> m_policyResult;
    std::optional<RefPtr<BrowsingWarning>> m_safeBrowsingWarning;
    std::optional<std::optional<NavigatingToAppBoundDomain>> m_isNavigatingToAppBoundDomain;
    bool m_doneWaitingForLinkDecorationFilteringData { false };
    Reply m_reply;
};

} // namespace WebKit
