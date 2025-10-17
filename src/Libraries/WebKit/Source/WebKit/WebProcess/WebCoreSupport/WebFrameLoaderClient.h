/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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

#include <WebCore/NavigationIdentifier.h>
#include <WebCore/SandboxFlags.h>
#include <optional>
#include <wtf/Function.h>
#include <wtf/Ref.h>
#include <wtf/Scope.h>

namespace WebCore {
enum class PolicyAction : uint8_t;
enum class PolicyDecisionMode;
enum class IsPerformingHTTPFallback : bool;
class FormState;
class Frame;
class HitTestResult;
class NavigationAction;
class ResourceRequest;
class ResourceResponse;
using FramePolicyFunction = CompletionHandler<void(PolicyAction)>;
}

namespace WebKit {

class WebFrame;
struct NavigationActionData;
struct WebsitePoliciesData;

class WebFrameLoaderClient {
public:
    WebFrame& webFrame() const { return m_frame.get(); }

    std::optional<NavigationActionData> navigationActionData(const WebCore::NavigationAction&, const WebCore::ResourceRequest&, const WebCore::ResourceResponse& redirectResponse, const String& clientRedirectSourceForHistory, std::optional<WebCore::NavigationIdentifier>, std::optional<WebCore::HitTestResult>&&, bool hasOpener, WebCore::IsPerformingHTTPFallback, WebCore::SandboxFlags) const;

    virtual void applyWebsitePolicies(WebsitePoliciesData&&) = 0;

    virtual ~WebFrameLoaderClient();

    ScopeExit<Function<void()>> takeFrameInvalidator() { return WTFMove(m_frameInvalidator); }

protected:
    WebFrameLoaderClient(Ref<WebFrame>&&, ScopeExit<Function<void()>>&& frameInvalidator);

    void dispatchDecidePolicyForNavigationAction(const WebCore::NavigationAction&, const WebCore::ResourceRequest&, const WebCore::ResourceResponse& redirectResponse, WebCore::FormState*, const String&, std::optional<WebCore::NavigationIdentifier>, std::optional<WebCore::HitTestResult>&&, bool, WebCore::IsPerformingHTTPFallback, WebCore::SandboxFlags, WebCore::PolicyDecisionMode, WebCore::FramePolicyFunction&&);
    void updateSandboxFlags(WebCore::SandboxFlags);
    void updateOpener(const WebCore::Frame&);

    Ref<WebFrame> m_frame;
    ScopeExit<Function<void()>> m_frameInvalidator;
};

}
