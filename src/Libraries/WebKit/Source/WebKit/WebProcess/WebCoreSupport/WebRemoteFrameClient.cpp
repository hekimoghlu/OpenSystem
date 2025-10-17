/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
#include "WebRemoteFrameClient.h"

#include "MessageSenderInlines.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <WebCore/FrameLoadRequest.h>
#include <WebCore/FrameTree.h>
#include <WebCore/HitTestResult.h>
#include <WebCore/PolicyChecker.h>
#include <WebCore/RemoteFrame.h>

namespace WebKit {
using namespace WebCore;

WebRemoteFrameClient::WebRemoteFrameClient(Ref<WebFrame>&& frame, ScopeExit<Function<void()>>&& frameInvalidator)
    : WebFrameLoaderClient(WTFMove(frame), WTFMove(frameInvalidator))
{
}

WebRemoteFrameClient::~WebRemoteFrameClient() = default;

void WebRemoteFrameClient::frameDetached()
{
    RefPtr coreFrame = m_frame->coreRemoteFrame();
    if (!coreFrame) {
        ASSERT_NOT_REACHED();
        return;
    }

    if (RefPtr parent = coreFrame->tree().parent()) {
        coreFrame->tree().detachFromParent();
        parent->tree().removeChild(*coreFrame);
    }
    m_frame->invalidate();
}

void WebRemoteFrameClient::sizeDidChange(IntSize size)
{
    m_frame->updateRemoteFrameSize(size);
}

void WebRemoteFrameClient::postMessageToRemote(FrameIdentifier source, const String& sourceOrigin, FrameIdentifier target, std::optional<SecurityOriginData> targetOrigin, const MessageWithMessagePorts& message)
{
    if (auto* page = m_frame->page())
        page->send(Messages::WebPageProxy::PostMessageToRemote(source, sourceOrigin, target, targetOrigin, message));
}

void WebRemoteFrameClient::changeLocation(FrameLoadRequest&& request)
{
    // FIXME: FrameLoadRequest and NavigationAction can probably be refactored to share more. <rdar://116202911>
    NavigationAction action(request.requester(), request.resourceRequest(), request.initiatedByMainFrame(), request.isRequestFromClientOrUserInput());
    // FIXME: action.request and request are probably duplicate information. <rdar://116203126>
    // FIXME: Get more parameters correct and add tests for each one. <rdar://116203354>
    dispatchDecidePolicyForNavigationAction(action, action.originalRequest(), ResourceResponse(), nullptr, { }, { }, { }, { }, IsPerformingHTTPFallback::No, { }, PolicyDecisionMode::Asynchronous, [protectedFrame = Ref { m_frame }, request = WTFMove(request)] (PolicyAction policyAction) mutable {
        // WebPage::loadRequest will make this load happen if needed.
        // FIXME: What if PolicyAction::Ignore is sent. Is everything in the right state? We probably need to make sure the load event still happens on the parent frame. <rdar://116203453>
    });
}

String WebRemoteFrameClient::renderTreeAsText(size_t baseIndent, OptionSet<RenderAsTextFlag> behavior)
{
    RefPtr page = m_frame->page();
    if (!page)
        return "Test Error - Missing page"_s;
    auto sendResult = page->sendSync(Messages::WebPageProxy::RenderTreeAsTextForTesting(m_frame->frameID(), baseIndent, behavior));
    if (!sendResult.succeeded())
        return "Test Error - sending WebPageProxy::RenderTreeAsTextForTesting failed"_s;
    auto [result] = sendResult.takeReply();
    return result;
}

String WebRemoteFrameClient::layerTreeAsText(size_t baseIndent, OptionSet<LayerTreeAsTextOptions> options)
{
    RefPtr page = m_frame->page();
    if (!page)
        return "Test Error - Missing page"_s;
    options.add(LayerTreeAsTextOptions::IncludeRootLayers);
    auto sendResult = page->sendSync(Messages::WebPageProxy::LayerTreeAsTextForTesting(m_frame->frameID(), baseIndent, options));
    if (!sendResult.succeeded())
        return "Test Error - sending WebPageProxy::LayerTreeAsTextForTesting failed"_s;
    auto [result] = sendResult.takeReply();
    return result;
}

void WebRemoteFrameClient::unbindRemoteAccessibilityFrames(int processIdentifier)
{
#if PLATFORM(COCOA)
    // Make sure AppKit system knows about our remote UI process status now.
    if (RefPtr page = m_frame->page())
        page->accessibilityManageRemoteElementStatus(false, processIdentifier);
#else
    UNUSED_PARAM(processIdentifier);
#endif
}

void WebRemoteFrameClient::updateRemoteFrameAccessibilityOffset(WebCore::FrameIdentifier frameID, WebCore::IntPoint offset)
{
    if (RefPtr page = m_frame->page())
        page->send(Messages::WebPageProxy::UpdateRemoteFrameAccessibilityOffset(frameID, offset));
}

void WebRemoteFrameClient::bindRemoteAccessibilityFrames(int processIdentifier, WebCore::FrameIdentifier frameID, Vector<uint8_t>&& dataToken, CompletionHandler<void(Vector<uint8_t>, int)>&& completionHandler)
{
    RefPtr page = m_frame->page();
    if (!page) {
        completionHandler({ }, 0);
        return;
    }

    auto sendResult = page->sendSync(Messages::WebPageProxy::BindRemoteAccessibilityFrames(processIdentifier, frameID, WTFMove(dataToken)));
    if (!sendResult.succeeded()) {
        completionHandler({ }, 0);
        return;
    }

    auto [resultToken, processIdentifierResult] = sendResult.takeReply();

#if PLATFORM(MAC)
    // Make sure AppKit system knows about our remote UI process status now.
    page->accessibilityManageRemoteElementStatus(true, processIdentifierResult);
#endif
    completionHandler(resultToken, processIdentifierResult);
}

void WebRemoteFrameClient::closePage()
{
    if (auto* page = m_frame->page())
        page->sendClose();
}

void WebRemoteFrameClient::focus()
{
    if (auto* page = m_frame->page())
        page->send(Messages::WebPageProxy::FocusRemoteFrame(m_frame->frameID()));
}

void WebRemoteFrameClient::unfocus()
{
    if (auto* page = m_frame->page())
        page->send(Messages::WebPageProxy::SetFocus(false));
}

void WebRemoteFrameClient::documentURLForConsoleLog(CompletionHandler<void(const URL&)>&& completionHandler)
{
    if (auto* page = m_frame->page())
        page->sendWithAsyncReply(Messages::WebPageProxy::DocumentURLForConsoleLog(m_frame->frameID()), WTFMove(completionHandler));
    else
        completionHandler({ });
}

void WebRemoteFrameClient::dispatchDecidePolicyForNavigationAction(const NavigationAction& navigationAction, const ResourceRequest& request, const ResourceResponse& redirectResponse,
    FormState* formState, const String& clientRedirectSourceForHistory, std::optional<WebCore::NavigationIdentifier> navigationID, std::optional<HitTestResult>&& hitTestResult, bool hasOpener, IsPerformingHTTPFallback isPerformingHTTPFallback, SandboxFlags sandboxFlags, PolicyDecisionMode policyDecisionMode, FramePolicyFunction&& function)
{
    WebFrameLoaderClient::dispatchDecidePolicyForNavigationAction(navigationAction, request, redirectResponse, formState, clientRedirectSourceForHistory, navigationID, WTFMove(hitTestResult), hasOpener, isPerformingHTTPFallback, sandboxFlags, policyDecisionMode, WTFMove(function));
}

void WebRemoteFrameClient::updateSandboxFlags(WebCore::SandboxFlags sandboxFlags)
{
    WebFrameLoaderClient::updateSandboxFlags(sandboxFlags);
}

void WebRemoteFrameClient::updateOpener(const WebCore::Frame& newOpener)
{
    WebFrameLoaderClient::updateOpener(newOpener);
}

void WebRemoteFrameClient::applyWebsitePolicies(WebsitePoliciesData&& websitePolicies)
{
    RefPtr coreFrame = m_frame->coreRemoteFrame();
    if (!coreFrame) {
        ASSERT_NOT_REACHED();
        return;
    }

    coreFrame->setCustomUserAgent(websitePolicies.customUserAgent);
    coreFrame->setCustomUserAgentAsSiteSpecificQuirks(websitePolicies.customUserAgentAsSiteSpecificQuirks);
    coreFrame->setAdvancedPrivacyProtections(websitePolicies.advancedPrivacyProtections);
    coreFrame->setCustomNavigatorPlatform(websitePolicies.customNavigatorPlatform);
}

void WebRemoteFrameClient::updateScrollingMode(ScrollbarMode scrollingMode)
{
    if (auto* page = m_frame->page())
        page->send(Messages::WebPageProxy::UpdateScrollingMode(m_frame->frameID(), scrollingMode));
}

}
