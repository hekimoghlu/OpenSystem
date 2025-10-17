/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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

#include "WebFrame.h"
#include "WebFrameLoaderClient.h"
#include <WebCore/MessageWithMessagePorts.h>
#include <WebCore/ProcessIdentifier.h>
#include <WebCore/RemoteFrameClient.h>
#include <WebCore/SecurityOriginData.h>

namespace WebKit {

class WebRemoteFrameClient final : public WebCore::RemoteFrameClient, public WebFrameLoaderClient {
public:
    explicit WebRemoteFrameClient(Ref<WebFrame>&&, ScopeExit<Function<void()>>&& frameInvalidator);
    ~WebRemoteFrameClient();

    void applyWebsitePolicies(WebsitePoliciesData&&) final;

private:
    void frameDetached() final;
    void sizeDidChange(WebCore::IntSize) final;
    void postMessageToRemote(WebCore::FrameIdentifier source, const String& sourceOrigin, WebCore::FrameIdentifier target, std::optional<WebCore::SecurityOriginData> targetOrigin, const WebCore::MessageWithMessagePorts&) final;
    void changeLocation(WebCore::FrameLoadRequest&&) final;
    String renderTreeAsText(size_t baseIndent, OptionSet<WebCore::RenderAsTextFlag>) final;
    String layerTreeAsText(size_t baseIndent, OptionSet<WebCore::LayerTreeAsTextOptions>) final;
    void bindRemoteAccessibilityFrames(int processIdentifier, WebCore::FrameIdentifier, Vector<uint8_t>&&, CompletionHandler<void(Vector<uint8_t>, int)>&&) final;
    void unbindRemoteAccessibilityFrames(int) final;
    void updateRemoteFrameAccessibilityOffset(WebCore::FrameIdentifier, WebCore::IntPoint) final;

    void closePage() final;
    void focus() final;
    void unfocus() final;
    void documentURLForConsoleLog(CompletionHandler<void(const URL&)>&&) final;
    void dispatchDecidePolicyForNavigationAction(const WebCore::NavigationAction&, const WebCore::ResourceRequest&, const WebCore::ResourceResponse& redirectResponse, WebCore::FormState*, const String& clientRedirectSourceForHistory, std::optional<WebCore::NavigationIdentifier>, std::optional<WebCore::HitTestResult>&&, bool hasOpener, WebCore::IsPerformingHTTPFallback, WebCore::SandboxFlags, WebCore::PolicyDecisionMode, WebCore::FramePolicyFunction&&) final;
    void updateSandboxFlags(WebCore::SandboxFlags) final;
    void updateOpener(const WebCore::Frame&) final;
    void updateScrollingMode(WebCore::ScrollbarMode scrollingMode) final;
};

}
