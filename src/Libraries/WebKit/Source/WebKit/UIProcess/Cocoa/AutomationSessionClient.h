/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#ifndef AutomationSessionClient_h
#define AutomationSessionClient_h

#import "WKFoundation.h"

#import "APIAutomationSessionClient.h"
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakObjCPtr.h>

@protocol _WKAutomationSessionDelegate;

namespace WebKit {

class AutomationSessionClient final : public API::AutomationSessionClient {
    WTF_MAKE_TZONE_ALLOCATED(AutomationSessionClient);
public:
    explicit AutomationSessionClient(id <_WKAutomationSessionDelegate>);

private:
    // From API::AutomationSessionClient
    void didDisconnectFromRemote(WebAutomationSession&) override;

    void requestNewPageWithOptions(WebAutomationSession&, API::AutomationSessionBrowsingContextOptions, CompletionHandler<void(WebPageProxy*)>&&) override;
    void requestSwitchToPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&, CompletionHandler<void()>&&) override;
    void requestHideWindowOfPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&, CompletionHandler<void()>&&) override;
    void requestRestoreWindowOfPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&, CompletionHandler<void()>&&) override;
    void requestMaximizeWindowOfPage(WebKit::WebAutomationSession&, WebKit::WebPageProxy&, CompletionHandler<void()>&&) override;

#if ENABLE(WK_WEB_EXTENSIONS_IN_WEBDRIVER)
    void loadWebExtensionWithOptions(WebKit::WebAutomationSession&, API::AutomationSessionWebExtensionResourceOptions, const String& resource, CompletionHandler<void(const String&)>&&) override;
    void unloadWebExtension(WebKit::WebAutomationSession&, const String& identifier, CompletionHandler<void(bool)>&&) override;
#endif

    bool isShowingJavaScriptDialogOnPage(WebAutomationSession&, WebPageProxy&) override;
    void dismissCurrentJavaScriptDialogOnPage(WebAutomationSession&, WebPageProxy&) override;
    void acceptCurrentJavaScriptDialogOnPage(WebAutomationSession&, WebPageProxy&) override;
    String messageOfCurrentJavaScriptDialogOnPage(WebAutomationSession&, WebPageProxy&) override;
    void setUserInputForCurrentJavaScriptPromptOnPage(WebAutomationSession&, WebPageProxy&, const String&) override;
    std::optional<API::AutomationSessionClient::JavaScriptDialogType> typeOfCurrentJavaScriptDialogOnPage(WebAutomationSession&, WebPageProxy&) override;
    API::AutomationSessionClient::BrowsingContextPresentation currentPresentationOfPage(WebAutomationSession&, WebPageProxy&) override;

    WeakObjCPtr<id <_WKAutomationSessionDelegate>> m_delegate;

    struct {
        bool didDisconnectFromRemote : 1;

        bool requestNewWebViewWithOptions : 1;
        bool requestSwitchToWebView : 1;
        bool requestHideWindowOfWebView : 1;
        bool requestRestoreWindowOfWebView : 1;
        bool requestMaximizeWindowOfWebView : 1;
        bool isShowingJavaScriptDialogForWebView : 1;
        bool dismissCurrentJavaScriptDialogForWebView : 1;
        bool acceptCurrentJavaScriptDialogForWebView : 1;
        bool messageOfCurrentJavaScriptDialogForWebView : 1;
        bool setUserInputForCurrentJavaScriptPromptForWebView : 1;
        bool typeOfCurrentJavaScriptDialogForWebView : 1;
        bool currentPresentationForWebView : 1;
        bool loadWebExtensionWithOptions : 1;
        bool unloadWebExtension : 1;
    } m_delegateMethods;
};

} // namespace WebKit

#endif // AutomationSessionClient_h
