/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 2, 2023.
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
#include "WebKitNavigationClient.h"

#include "APINavigationAction.h"
#include "APINavigationClient.h"
#include "FrameInfoData.h"
#include "WebKitBackForwardListPrivate.h"
#include "WebKitDownloadPrivate.h"
#include "WebKitNavigationPolicyDecisionPrivate.h"
#include "WebKitPrivate.h"
#include "WebKitResponsePolicyDecisionPrivate.h"
#include "WebKitURIResponsePrivate.h"
#include "WebKitWebContextPrivate.h"
#include "WebKitWebViewPrivate.h"
#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/CString.h>

#if ENABLE(2022_GLIB_API)
#include "WebKitNetworkSessionPrivate.h"
#endif

using namespace WebKit;
using namespace WebCore;

class NavigationClient : public API::NavigationClient {
public:
    explicit NavigationClient(WebKitWebView* webView)
        : m_webView(webView)
    {
    }

private:
    void didStartProvisionalNavigation(WebPageProxy&, const ResourceRequest&, API::Navigation*, API::Object* /* userData */) override
    {
        webkitWebViewLoadChanged(m_webView, WEBKIT_LOAD_STARTED);
    }

    void didReceiveServerRedirectForProvisionalNavigation(WebPageProxy&, API::Navigation*, API::Object* /* userData */) override
    {
        webkitWebViewLoadChanged(m_webView, WEBKIT_LOAD_REDIRECTED);
    }

    void didFailProvisionalNavigationWithError(WebPageProxy&, FrameInfoData&& frameInfo, API::Navigation*, const URL&, const ResourceError& resourceError, API::Object* /* userData */) override
    {
        if (!frameInfo.isMainFrame)
            return;
        GUniquePtr<GError> error(g_error_new_literal(g_quark_from_string(resourceError.domain().utf8().data()),
            toWebKitError(resourceError.errorCode()), resourceError.localizedDescription().utf8().data()));
        if (resourceError.tlsErrors()) {
            webkitWebViewLoadFailedWithTLSErrors(m_webView, resourceError.failingURL().string().utf8().data(), error.get(),
                static_cast<GTlsCertificateFlags>(resourceError.tlsErrors()), resourceError.certificate());
        } else
            webkitWebViewLoadFailed(m_webView, WEBKIT_LOAD_STARTED, resourceError.failingURL().string().utf8().data(), error.get());
    }

    void didCommitNavigation(WebPageProxy&, API::Navigation*, API::Object* /* userData */) override
    {
        webkitWebViewLoadChanged(m_webView, WEBKIT_LOAD_COMMITTED);
    }

    void didFinishNavigation(WebPageProxy&, API::Navigation*, API::Object* /* userData */) override
    {
        webkitWebViewLoadChanged(m_webView, WEBKIT_LOAD_FINISHED);
    }

    void didFailNavigationWithError(WebPageProxy&, const FrameInfoData& frameInfo, API::Navigation*, const URL&, const ResourceError& resourceError, API::Object* /* userData */) override
    {
        if (!frameInfo.isMainFrame)
            return;
        GUniquePtr<GError> error(g_error_new_literal(g_quark_from_string(resourceError.domain().utf8().data()),
            toWebKitError(resourceError.errorCode()), resourceError.localizedDescription().utf8().data()));
        webkitWebViewLoadFailed(m_webView, WEBKIT_LOAD_COMMITTED, resourceError.failingURL().string().utf8().data(), error.get());
    }

    void didDisplayInsecureContent(WebPageProxy&, API::Object* /* userData */) override
    {
        webkitWebViewInsecureContentDetected(m_webView, WEBKIT_INSECURE_CONTENT_DISPLAYED);
    }

    void didRunInsecureContent(WebPageProxy&, API::Object* /* userData */) override
    {
        webkitWebViewInsecureContentDetected(m_webView, WEBKIT_INSECURE_CONTENT_RUN);
    }

    bool didChangeBackForwardList(WebPageProxy&, WebBackForwardListItem* addedItem, const Vector<Ref<WebBackForwardListItem>>& removedItems) override
    {
        webkitBackForwardListChanged(webkit_web_view_get_back_forward_list(m_webView), addedItem, removedItems);
        return true;
    }

    void didReceiveAuthenticationChallenge(WebPageProxy&, AuthenticationChallengeProxy& authenticationChallenge) override
    {
        webkitWebViewHandleAuthenticationChallenge(m_webView, &authenticationChallenge);
    }

    bool processDidTerminate(WebPageProxy&, ProcessTerminationReason reason) override
    {
        switch (reason) {
        case ProcessTerminationReason::Crash:
        case ProcessTerminationReason::NonMainFrameWebContentProcessCrash:
            webkitWebViewWebProcessTerminated(m_webView, WEBKIT_WEB_PROCESS_CRASHED);
            return true;
        case ProcessTerminationReason::ExceededMemoryLimit:
            webkitWebViewWebProcessTerminated(m_webView, WEBKIT_WEB_PROCESS_EXCEEDED_MEMORY_LIMIT);
            return true;
        case ProcessTerminationReason::RequestedByClient:
            webkitWebViewWebProcessTerminated(m_webView, WEBKIT_WEB_PROCESS_TERMINATED_BY_API);
            return true;
        case ProcessTerminationReason::ExceededCPULimit:
        case ProcessTerminationReason::RequestedByNetworkProcess:
        case ProcessTerminationReason::NavigationSwap:
        case ProcessTerminationReason::RequestedByGPUProcess:
        case ProcessTerminationReason::RequestedByModelProcess:
        case ProcessTerminationReason::ExceededProcessCountLimit:
        case ProcessTerminationReason::IdleExit:
        case ProcessTerminationReason::Unresponsive:
        case ProcessTerminationReason::GPUProcessCrashedTooManyTimes:
        case ProcessTerminationReason::ModelProcessCrashedTooManyTimes:
            break;
        }
        return false;
    }

    void processDidBecomeResponsive(WebKit::WebPageProxy&) override
    {
        webkitWebViewSetIsWebProcessResponsive(m_webView, true);
    }

    void processDidBecomeUnresponsive(WebKit::WebPageProxy&) override
    {
        webkitWebViewSetIsWebProcessResponsive(m_webView, false);
    }

    void decidePolicyForNavigationAction(WebPageProxy&, Ref<API::NavigationAction>&& navigationAction, Ref<WebFramePolicyListenerProxy>&& listener) override
    {
        WebKitPolicyDecisionType decisionType = navigationAction->targetFrame() ? WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION : WEBKIT_POLICY_DECISION_TYPE_NEW_WINDOW_ACTION;
        GRefPtr<WebKitPolicyDecision> decision = adoptGRef(webkitNavigationPolicyDecisionCreate(WTFMove(navigationAction), WTFMove(listener)));
        webkitWebViewMakePolicyDecision(m_webView, decisionType, decision.get());
    }

    void decidePolicyForNavigationResponse(WebPageProxy&, Ref<API::NavigationResponse>&& navigationResponse, Ref<WebFramePolicyListenerProxy>&& listener) override
    {
        GRefPtr<WebKitPolicyDecision> decision = adoptGRef(webkitResponsePolicyDecisionCreate(WTFMove(navigationResponse), WTFMove(listener)));
        webkitWebViewMakePolicyDecision(m_webView, WEBKIT_POLICY_DECISION_TYPE_RESPONSE, decision.get());
    }

    void navigationActionDidBecomeDownload(WebPageProxy&, API::NavigationAction&, DownloadProxy& downloadProxy) override
    {
        auto download = webkitDownloadCreate(downloadProxy, m_webView);
#if ENABLE(2022_GLIB_API)
        webkitNetworkSessionDownloadStarted(webkit_web_view_get_network_session(m_webView), download.get());
#else
        webkitWebContextDownloadStarted(webkit_web_view_get_context(m_webView), download.get());
#endif
    }

    void navigationResponseDidBecomeDownload(WebPageProxy&, API::NavigationResponse&, DownloadProxy& downloadProxy) override
    {
        auto download = webkitDownloadCreate(downloadProxy, m_webView);
#if ENABLE(2022_GLIB_API)
        webkitNetworkSessionDownloadStarted(webkit_web_view_get_network_session(m_webView), download.get());
#else
        webkitWebContextDownloadStarted(webkit_web_view_get_context(m_webView), download.get());
#endif
    }

    void contextMenuDidCreateDownload(WebPageProxy&, DownloadProxy& downloadProxy) override
    {
        auto download = webkitDownloadCreate(downloadProxy, m_webView);
#if ENABLE(2022_GLIB_API)
        webkitNetworkSessionDownloadStarted(webkit_web_view_get_network_session(m_webView), download.get());
#else
        webkitWebContextDownloadStarted(webkit_web_view_get_context(m_webView), download.get());
#endif
    }

    WebKitWebView* m_webView;
};

void attachNavigationClientToView(WebKitWebView* webView)
{
    webkitWebViewGetPage(webView).setNavigationClient(makeUniqueRef<NavigationClient>(webView));
}

