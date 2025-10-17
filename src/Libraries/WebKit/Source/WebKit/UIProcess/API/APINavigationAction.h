/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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

#include "APIFrameInfo.h"
#include "APINavigation.h"
#include "APIObject.h"
#include "APIUserInitiatedAction.h"
#include "NavigationActionData.h"
#include "WebHitTestResultData.h"
#include <WebCore/ResourceRequest.h>
#include <wtf/URL.h>

namespace API {

class NavigationAction final : public ObjectImpl<Object::Type::NavigationAction> {
public:
    template<typename... Args> static Ref<NavigationAction> create(Args&&... args)
    {
        return adoptRef(*new NavigationAction(std::forward<Args>(args)...));
    }

    FrameInfo* sourceFrame() const { return m_sourceFrame.get(); }
    FrameInfo* targetFrame() const { return m_targetFrame.get(); }
    const WTF::String& targetFrameName() const { return m_targetFrameName; }

    const WebCore::ResourceRequest& request() const { return m_request; }
    const WTF::URL& originalURL() const { return !m_originalURL.isNull() ? m_originalURL : m_request.url(); }

    WebCore::NavigationType navigationType() const { return m_navigationActionData.navigationType; }
    OptionSet<WebKit::WebEventModifier> modifiers() const { return m_navigationActionData.modifiers; }
    WebKit::WebMouseEventButton mouseButton() const { return m_navigationActionData.mouseButton; }
    WebKit::WebMouseEventSyntheticClickType syntheticClickType() const { return m_navigationActionData.syntheticClickType; }
#if PLATFORM(MAC) || HAVE(UIKIT_WITH_MOUSE_SUPPORT)
    const std::optional<WebKit::WebHitTestResultData>& webHitTestResultData() const { return m_navigationActionData.webHitTestResultData; }
#endif
    WebCore::FloatPoint clickLocationInRootViewCoordinates() const { return m_navigationActionData.clickLocationInRootViewCoordinates; }
    bool canHandleRequest() const { return m_navigationActionData.canHandleRequest; }
    bool shouldOpenExternalSchemes() const { return m_navigationActionData.shouldOpenExternalURLsPolicy == WebCore::ShouldOpenExternalURLsPolicy::ShouldAllow || m_navigationActionData.shouldOpenExternalURLsPolicy == WebCore::ShouldOpenExternalURLsPolicy::ShouldAllowExternalSchemesButNotAppLinks; }
    bool shouldOpenAppLinks() const { return m_shouldOpenAppLinks && m_navigationActionData.shouldOpenExternalURLsPolicy == WebCore::ShouldOpenExternalURLsPolicy::ShouldAllow; }
    bool shouldPerformDownload() const { return !m_navigationActionData.downloadAttribute.isNull(); }
    bool isRedirect() const { return !m_navigationActionData.redirectResponse.isNull(); }
    bool hasOpener() const { return m_navigationActionData.hasOpener; }
    WebCore::ShouldOpenExternalURLsPolicy shouldOpenExternalURLsPolicy() const { return m_navigationActionData.shouldOpenExternalURLsPolicy; }

    bool isProcessingUserGesture() const { return m_userInitiatedAction; }
    bool isProcessingUnconsumedUserGesture() const { return m_userInitiatedAction && !m_userInitiatedAction->consumed(); }
    UserInitiatedAction* userInitiatedAction() const { return m_userInitiatedAction.get(); }

    Navigation* mainFrameNavigation() const { return m_mainFrameNavigation.get(); }

#if HAVE(APP_SSO)
    bool shouldPerformSOAuthorization() { return m_shouldPerformSOAuthorization; }
    void unsetShouldPerformSOAuthorization() { m_shouldPerformSOAuthorization = false; }
#endif

    const WebKit::NavigationActionData& data() const { return m_navigationActionData; }
    std::optional<WebCore::FrameIdentifier> mainFrameIDBeforeNavigationActionDecision() { return m_mainFrameIDBeforeNavigationDecision; }
    
private:
    NavigationAction(WebKit::NavigationActionData&& navigationActionData, API::FrameInfo* sourceFrame, API::FrameInfo* targetFrame, const WTF::String& targetFrameName, WebCore::ResourceRequest&& request, const WTF::URL& originalURL, bool shouldOpenAppLinks, RefPtr<UserInitiatedAction>&& userInitiatedAction, API::Navigation* mainFrameNavigation, std::optional<WebCore::FrameIdentifier> mainFrameIDBeforeNavigationActionDecision)
        : m_sourceFrame(sourceFrame)
        , m_targetFrame(targetFrame)
        , m_targetFrameName(targetFrameName)
        , m_request(WTFMove(request))
        , m_originalURL(originalURL)
        , m_shouldOpenAppLinks(shouldOpenAppLinks)
        , m_userInitiatedAction(WTFMove(userInitiatedAction))
        , m_navigationActionData(WTFMove(navigationActionData))
        , m_mainFrameNavigation(mainFrameNavigation)
        , m_mainFrameIDBeforeNavigationDecision(mainFrameIDBeforeNavigationActionDecision)
    {
    }

    NavigationAction(WebKit::NavigationActionData&& navigationActionData, API::FrameInfo* sourceFrame, API::FrameInfo* targetFrame, const WTF::String& targetFrameName, WebCore::ResourceRequest&& request, const WTF::URL& originalURL, bool shouldOpenAppLinks, RefPtr<UserInitiatedAction>&& userInitiatedAction)
        : NavigationAction(WTFMove(navigationActionData), sourceFrame, targetFrame, targetFrameName, WTFMove(request), originalURL, shouldOpenAppLinks, WTFMove(userInitiatedAction), nullptr, std::nullopt)
    {
    }

    RefPtr<FrameInfo> m_sourceFrame;
    RefPtr<FrameInfo> m_targetFrame;
    WTF::String m_targetFrameName;

    WebCore::ResourceRequest m_request;
    WTF::URL m_originalURL;

    bool m_shouldOpenAppLinks;
#if HAVE(APP_SSO)
    bool m_shouldPerformSOAuthorization { true };
#endif

    RefPtr<UserInitiatedAction> m_userInitiatedAction;

    const WebKit::NavigationActionData m_navigationActionData;
    RefPtr<Navigation> m_mainFrameNavigation;
    std::optional<WebCore::FrameIdentifier> m_mainFrameIDBeforeNavigationDecision;
};

} // namespace API
