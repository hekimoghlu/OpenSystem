/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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

#include "APIData.h"
#include "APIString.h"
#include "AuthenticationChallengeDisposition.h"
#include "AuthenticationChallengeProxy.h"
#include "AuthenticationDecisionListener.h"
#include "ProcessTerminationReason.h"
#include "SameDocumentNavigationType.h"
#include "WebFramePolicyListenerProxy.h"
#include "WebsitePoliciesData.h"
#include <WebCore/FrameLoaderTypes.h>
#include <WebCore/LayoutMilestone.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>

#if HAVE(APP_SSO)
#include "SOAuthorizationLoadPolicy.h"
#endif

namespace WebCore {
struct ContentRuleListResults;
class ResourceError;
class ResourceRequest;
class ResourceResponse;
class FragmentedSharedBuffer;
class SecurityOriginData;
}

namespace WebKit {
class AuthenticationChallengeProxy;
class DownloadProxy;
class WebBackForwardListItem;
class WebFramePolicyListenerProxy;
class WebFrameProxy;
class WebPageProxy;
class WebProtectionSpace;
struct FrameInfoData;
struct NavigationActionData;
struct WebNavigationDataStore;
}

namespace API {

class Dictionary;
class Navigation;
class NavigationAction;
class NavigationResponse;
class Object;

class NavigationClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(NavigationClient);
public:
    virtual ~NavigationClient() { }

    virtual void didStartProvisionalNavigation(WebKit::WebPageProxy&, const WebCore::ResourceRequest&, Navigation*, Object*) { }
    virtual void didStartProvisionalLoadForFrame(WebKit::WebPageProxy&, WebCore::ResourceRequest&&, WebKit::FrameInfoData&&) { }
    virtual void didReceiveServerRedirectForProvisionalNavigation(WebKit::WebPageProxy&, Navigation*, Object*) { }
    virtual void willPerformClientRedirect(WebKit::WebPageProxy&, const WTF::String& destinationURL, double) { }
    virtual void didPerformClientRedirect(WebKit::WebPageProxy&, const WTF::String& sourceURL, const WTF::String& destinationURL) { }
    virtual void didCancelClientRedirect(WebKit::WebPageProxy&) { }
    virtual void didFailProvisionalNavigationWithError(WebKit::WebPageProxy&, WebKit::FrameInfoData&&, Navigation*, const WTF::URL&, const WebCore::ResourceError&, Object*) { }
    virtual void didFailProvisionalLoadWithErrorForFrame(WebKit::WebPageProxy&, WebCore::ResourceRequest&&, const WebCore::ResourceError&, WebKit::FrameInfoData&&) { }
    virtual void didCommitNavigation(WebKit::WebPageProxy&, Navigation*, Object*) { }
    virtual void didCommitLoadForFrame(WebKit::WebPageProxy&, WebCore::ResourceRequest&&, WebKit::FrameInfoData&&) { }
    virtual void didFinishDocumentLoad(WebKit::WebPageProxy&, Navigation*, Object*) { }
    virtual void didFinishNavigation(WebKit::WebPageProxy&, Navigation*, Object*) { }
    virtual void didFinishLoadForFrame(WebKit::WebPageProxy&, WebCore::ResourceRequest&&, WebKit::FrameInfoData&&) { }
    virtual void didBlockLoadToKnownTracker(WebKit::WebPageProxy&, const WTF::URL&) { }
    virtual void didFailNavigationWithError(WebKit::WebPageProxy&, const WebKit::FrameInfoData&, Navigation*, const WTF::URL&, const WebCore::ResourceError&, Object*) { }
    virtual void didFailLoadWithErrorForFrame(WebKit::WebPageProxy&, WebCore::ResourceRequest&&, const WebCore::ResourceError&, WebKit::FrameInfoData&&) { }
    virtual void didSameDocumentNavigation(WebKit::WebPageProxy&, Navigation*, WebKit::SameDocumentNavigationType, Object*) { }
    virtual void didApplyLinkDecorationFiltering(WebKit::WebPageProxy&, const WTF::URL&, const WTF::URL&) { }
    virtual void didPromptForStorageAccess(WebKit::WebPageProxy&, const WTF::String& topFrameDomain, const WTF::String& subFrameDomain, bool hasQuirk) { }

    virtual void didDisplayInsecureContent(WebKit::WebPageProxy&, API::Object*) { }
    virtual void didRunInsecureContent(WebKit::WebPageProxy&, API::Object*) { }

    virtual void renderingProgressDidChange(WebKit::WebPageProxy&, OptionSet<WebCore::LayoutMilestone>) { }

    virtual void navigationResponseDidBecomeDownload(WebKit::WebPageProxy&, NavigationResponse&, WebKit::DownloadProxy&) { }
    virtual void navigationActionDidBecomeDownload(WebKit::WebPageProxy&, NavigationAction&, WebKit::DownloadProxy&) { }
    virtual void contextMenuDidCreateDownload(WebKit::WebPageProxy&, WebKit::DownloadProxy&) { }

    virtual void didReceiveAuthenticationChallenge(WebKit::WebPageProxy&, WebKit::AuthenticationChallengeProxy& challenge) { challenge.listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::PerformDefaultHandling); }
    virtual void shouldAllowLegacyTLS(WebKit::WebPageProxy&, WebKit::AuthenticationChallengeProxy&, CompletionHandler<void(bool)>&& completionHandler) { completionHandler(true); }
    virtual void didNegotiateModernTLS(const WTF::URL&) { }
    virtual bool shouldBypassContentModeSafeguards() const { return false; }

    // FIXME: These function should not be part of this client.
    virtual bool processDidTerminate(WebKit::WebPageProxy&, WebKit::ProcessTerminationReason) { return false; }
    virtual void processDidBecomeResponsive(WebKit::WebPageProxy&) { }
    virtual void processDidBecomeUnresponsive(WebKit::WebPageProxy&) { }

    virtual void legacyWebCryptoMasterKey(WebKit::WebPageProxy&, CompletionHandler<void(std::optional<Vector<uint8_t>>&&)>&& completionHandler) { completionHandler(std::nullopt); }

#if USE(QUICK_LOOK)
    virtual void didStartLoadForQuickLookDocumentInMainFrame(const WTF::String& fileName, const WTF::String& uti) { }
    virtual void didFinishLoadForQuickLookDocumentInMainFrame(const WebCore::FragmentedSharedBuffer&) { }
#endif

    virtual void decidePolicyForNavigationAction(WebKit::WebPageProxy&, Ref<NavigationAction>&&, Ref<WebKit::WebFramePolicyListenerProxy>&& listener)
    {
        listener->use();
    }

    virtual void decidePolicyForNavigationResponse(WebKit::WebPageProxy&, Ref<NavigationResponse>&&, Ref<WebKit::WebFramePolicyListenerProxy>&& listener)
    {
        listener->use();
    }
    
    virtual void contentRuleListNotification(WebKit::WebPageProxy&, WTF::URL&&, WebCore::ContentRuleListResults&&) { };

    virtual void shouldGoToBackForwardListItem(WebKit::WebPageProxy&, WebKit::WebBackForwardListItem&, bool inBackForwardCache, CompletionHandler<void(bool)>&& completionHandler)
    {
        completionHandler(true);
    }

    virtual void didBeginNavigationGesture(WebKit::WebPageProxy&) { }
    virtual void willEndNavigationGesture(WebKit::WebPageProxy&, bool willNavigate, WebKit::WebBackForwardListItem&) { }
    virtual void didEndNavigationGesture(WebKit::WebPageProxy&, bool willNavigate, WebKit::WebBackForwardListItem&) { }
    virtual void didRemoveNavigationGestureSnapshot(WebKit::WebPageProxy&) { }
    virtual bool didChangeBackForwardList(WebKit::WebPageProxy&, WebKit::WebBackForwardListItem*, const Vector<Ref<WebKit::WebBackForwardListItem>>&) { return false; }

#if HAVE(APP_SSO)
    virtual void decidePolicyForSOAuthorizationLoad(WebKit::WebPageProxy&, WebKit::SOAuthorizationLoadPolicy currentSOAuthorizationLoadPolicy, const WTF::String&, CompletionHandler<void(WebKit::SOAuthorizationLoadPolicy)>&& completionHandler)
    {
        completionHandler(currentSOAuthorizationLoadPolicy);
    }
#endif
};

} // namespace API
