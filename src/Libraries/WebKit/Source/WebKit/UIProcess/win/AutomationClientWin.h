/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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

#include "WebProcessPool.h"

#if ENABLE(REMOTE_INSPECTOR)
#include "APIAutomationClient.h"
#include "APIAutomationSessionClient.h"
#include "APIUIClient.h"
#include "WebView.h"
#include <JavaScriptCore/RemoteInspectorServer.h>
#endif

namespace WebKit {

#if ENABLE(REMOTE_INSPECTOR)

class AutomationSessionClient final : public API::AutomationSessionClient {
public:
    explicit AutomationSessionClient(const String&, const Inspector::RemoteInspector::Client::SessionCapabilities&);

    String sessionIdentifier() const override { return m_sessionIdentifier; }

    void requestNewPageWithOptions(WebKit::WebAutomationSession&, API::AutomationSessionBrowsingContextOptions, CompletionHandler<void(WebKit::WebPageProxy*)>&&) override;
    void didDisconnectFromRemote(WebKit::WebAutomationSession&) override;

    void retainWebView(Ref<WebView>&&);
    void releaseWebView(WebPageProxy*);

private:
    String m_sessionIdentifier;
    Inspector::RemoteInspector::Client::SessionCapabilities m_capabilities { };

    static void close(WKPageRef, const void*);

    static void didReceiveAuthenticationChallenge(WKPageRef, WKAuthenticationChallengeRef, const void*);
    void didReceiveAuthenticationChallenge(WKPageRef, WKAuthenticationChallengeRef);

    HashSet<Ref<WebView>> m_webViews;
};

class AutomationClient final : public API::AutomationClient, Inspector::RemoteInspector::Client {
public:
    explicit AutomationClient(WebProcessPool&);
    ~AutomationClient();

private:
    // API::AutomationClient
    bool allowsRemoteAutomation(WebKit::WebProcessPool*) final { return remoteAutomationAllowed(); }
    void didRequestAutomationSession(WebKit::WebProcessPool*, const String& sessionIdentifier) final { }

    // RemoteInspector::Client
    bool remoteAutomationAllowed() const override { return true; }

    // FIXME: should use valid value
    String browserName() const override { return "MiniBrowser"_s; }
    String browserVersion() const override { return "1.0"_s; }

    RefPtr<WebProcessPool> protectedProcessPool() const;

    void requestAutomationSession(const String&, const Inspector::RemoteInspector::Client::SessionCapabilities&) override;
    void closeAutomationSession() override;

    WeakPtr<WebProcessPool> m_processPool;
};

#endif

} // namespace WebKit
