/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
#include "AutomationClientWin.h"

#if ENABLE(REMOTE_INSPECTOR)
#include "APIPageConfiguration.h"
#include "WKAPICast.h"
#include "WebAutomationSession.h"
#include "WebPageProxy.h"
#include <WebKit/WKAuthenticationChallenge.h>
#include <WebKit/WKAuthenticationDecisionListener.h>
#include <WebKit/WKCredential.h>
#include <WebKit/WKRetainPtr.h>
#include <WebKit/WKString.h>
#include <wtf/RunLoop.h>
#endif

namespace WebKit {

#if ENABLE(REMOTE_INSPECTOR)

// AutomationSessionClient
AutomationSessionClient::AutomationSessionClient(const String& sessionIdentifier, const Inspector::RemoteInspector::Client::SessionCapabilities& capabilities)
    : m_sessionIdentifier(sessionIdentifier)
    , m_capabilities(capabilities)
{
}

void AutomationSessionClient::close(WKPageRef pageRef, const void* clientInfo)
{
    auto page = WebKit::toImpl(pageRef);
    page->setControlledByAutomation(false);

    auto sessionClient = static_cast<AutomationSessionClient*>(const_cast<void*>(clientInfo));
    sessionClient->releaseWebView(page);
}

void AutomationSessionClient::didReceiveAuthenticationChallenge(WKPageRef page, WKAuthenticationChallengeRef authenticationChallenge, const void *clientInfo)
{
    static_cast<AutomationSessionClient*>(const_cast<void*>(clientInfo))->didReceiveAuthenticationChallenge(page, authenticationChallenge);
}

void AutomationSessionClient::didReceiveAuthenticationChallenge(WKPageRef page, WKAuthenticationChallengeRef authenticationChallenge)
{
    auto decisionListener = WKAuthenticationChallengeGetDecisionListener(authenticationChallenge);
    if (m_capabilities.acceptInsecureCertificates) {
        auto username = adoptWK(WKStringCreateWithUTF8CString("accept server trust"));
        auto password = adoptWK(WKStringCreateWithUTF8CString(""));
        auto credential = adoptWK(WKCredentialCreate(username.get(), password.get(), kWKCredentialPersistenceNone));
        WKAuthenticationDecisionListenerUseCredential(decisionListener, credential.get());
    } else
        WKAuthenticationDecisionListenerRejectProtectionSpaceAndContinue(decisionListener);
}

void AutomationSessionClient::requestNewPageWithOptions(WebKit::WebAutomationSession& session, API::AutomationSessionBrowsingContextOptions options, CompletionHandler<void(WebKit::WebPageProxy*)>&& completionHandler)
{
    auto pageConfiguration = API::PageConfiguration::create();
    pageConfiguration->setProcessPool(session.protectedProcessPool());

    RECT r { };
    Ref newWindow = WebView::create(r, pageConfiguration, 0);

    auto newPage = newWindow->page();
    newPage->setControlledByAutomation(true);

    WKPageUIClientV0 uiClient = { };
    uiClient.base.version = 0;
    uiClient.base.clientInfo = this;
    uiClient.close = close;
    WKPageSetPageUIClient(toAPI(newPage), &uiClient.base);

    WKPageNavigationClientV0 navigationClient = { };
    navigationClient.base.version = 0;
    navigationClient.base.clientInfo = this;
    navigationClient.didReceiveAuthenticationChallenge = didReceiveAuthenticationChallenge;
    WKPageSetPageNavigationClient(toAPI(newPage), &navigationClient.base);

    retainWebView(WTFMove(newWindow));

    completionHandler(newPage);
}

void AutomationSessionClient::didDisconnectFromRemote(WebKit::WebAutomationSession& session)
{
    session.setClient(nullptr);

    RunLoop::main().dispatch([&session] {
        auto processPool = session.protectedProcessPool();
        if (processPool) {
            processPool->setAutomationSession(nullptr);
            processPool->setPagesControlledByAutomation(false);
        }
    });
}

void AutomationSessionClient::retainWebView(Ref<WebView>&& webView)
{
    m_webViews.add(WTFMove(webView));
}

void AutomationSessionClient::releaseWebView(WebPageProxy* page)
{
    m_webViews.removeIf([&](auto& view) {
        if (view->page() == page) {
            view->close();
            return true;
        }
        return false;
    });
}

// AutomationClient
AutomationClient::AutomationClient(WebProcessPool& processPool)
    : m_processPool(processPool)
{
    Inspector::RemoteInspector::singleton().setClient(this);
}

AutomationClient::~AutomationClient()
{
    Inspector::RemoteInspector::singleton().setClient(nullptr);
}

RefPtr<WebProcessPool> AutomationClient::protectedProcessPool() const
{
    if (RefPtr processPool = m_processPool.get())
        return processPool;

    return nullptr;
}

void AutomationClient::requestAutomationSession(const String& sessionIdentifier, const Inspector::RemoteInspector::Client::SessionCapabilities& capabilities)
{
    ASSERT(isMainRunLoop());

    auto session = adoptRef(new WebAutomationSession());
    session->setSessionIdentifier(sessionIdentifier);
    session->setClient(WTF::makeUnique<AutomationSessionClient>(sessionIdentifier, capabilities));
    m_processPool->setAutomationSession(WTFMove(session));
}

void AutomationClient::closeAutomationSession()
{
    RunLoop::main().dispatch([this] {
        auto processPool = protectedProcessPool();
        if (!processPool || !processPool->automationSession())
            return;

        processPool->automationSession()->setClient(nullptr);
        processPool->setAutomationSession(nullptr);
    });
}

#endif

} // namespace WebKit
