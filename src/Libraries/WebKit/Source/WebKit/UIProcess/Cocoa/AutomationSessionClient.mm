/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#import "config.h"
#import "AutomationSessionClient.h"

#import "WKSharedAPICast.h"
#import "WKWebViewInternal.h"
#import "WebAutomationSession.h"
#import "WebPageProxy.h"
#import "_WKAutomationSessionDelegate.h"
#import "_WKAutomationSessionInternal.h"
#import <wtf/BlockPtr.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AutomationSessionClient);

AutomationSessionClient::AutomationSessionClient(id <_WKAutomationSessionDelegate> delegate)
    : m_delegate(delegate)
{
    m_delegateMethods.didDisconnectFromRemote = [delegate respondsToSelector:@selector(_automationSessionDidDisconnectFromRemote:)];

    m_delegateMethods.requestNewWebViewWithOptions = [delegate respondsToSelector:@selector(_automationSession:requestNewWebViewWithOptions:completionHandler:)];
    m_delegateMethods.requestSwitchToWebView = [delegate respondsToSelector:@selector(_automationSession:requestSwitchToWebView:completionHandler:)];
    m_delegateMethods.requestHideWindowOfWebView = [delegate respondsToSelector:@selector(_automationSession:requestHideWindowOfWebView:completionHandler:)];
    m_delegateMethods.requestRestoreWindowOfWebView = [delegate respondsToSelector:@selector(_automationSession:requestRestoreWindowOfWebView:completionHandler:)];
    m_delegateMethods.requestMaximizeWindowOfWebView = [delegate respondsToSelector:@selector(_automationSession:requestMaximizeWindowOfWebView:completionHandler:)];
    m_delegateMethods.isShowingJavaScriptDialogForWebView = [delegate respondsToSelector:@selector(_automationSession:isShowingJavaScriptDialogForWebView:)];
    m_delegateMethods.dismissCurrentJavaScriptDialogForWebView = [delegate respondsToSelector:@selector(_automationSession:dismissCurrentJavaScriptDialogForWebView:)];
    m_delegateMethods.acceptCurrentJavaScriptDialogForWebView = [delegate respondsToSelector:@selector(_automationSession:acceptCurrentJavaScriptDialogForWebView:)];
    m_delegateMethods.messageOfCurrentJavaScriptDialogForWebView = [delegate respondsToSelector:@selector(_automationSession:messageOfCurrentJavaScriptDialogForWebView:)];
    m_delegateMethods.setUserInputForCurrentJavaScriptPromptForWebView = [delegate respondsToSelector:@selector(_automationSession:setUserInput:forCurrentJavaScriptDialogForWebView:)];
    m_delegateMethods.typeOfCurrentJavaScriptDialogForWebView = [delegate respondsToSelector:@selector(_automationSession:typeOfCurrentJavaScriptDialogForWebView:)];
    m_delegateMethods.currentPresentationForWebView = [delegate respondsToSelector:@selector(_automationSession:currentPresentationForWebView:)];
#if ENABLE(WK_WEB_EXTENSIONS_IN_WEBDRIVER)
    m_delegateMethods.loadWebExtensionWithOptions = [delegate respondsToSelector:@selector(_automationSession:loadWebExtensionWithOptions:resource:completionHandler:)];
    m_delegateMethods.unloadWebExtension = [delegate respondsToSelector:@selector(_automationSession:unloadWebExtensionWithIdentifier:completionHandler:)];
#endif
}

void AutomationSessionClient::didDisconnectFromRemote(WebAutomationSession& session)
{
    if (m_delegateMethods.didDisconnectFromRemote)
        [m_delegate.get() _automationSessionDidDisconnectFromRemote:wrapper(session)];
}

static inline _WKAutomationSessionBrowsingContextOptions toAPI(API::AutomationSessionBrowsingContextOptions options)
{
    uint16_t wkOptions = 0;

    if (options & API::AutomationSessionBrowsingContextOptionsPreferNewTab)
        wkOptions |= _WKAutomationSessionBrowsingContextOptionsPreferNewTab;

    return static_cast<_WKAutomationSessionBrowsingContextOptions>(wkOptions);
}

#if ENABLE(WK_WEB_EXTENSIONS_IN_WEBDRIVER)
static inline _WKAutomationSessionWebExtensionResourceOptions toAPI(API::AutomationSessionWebExtensionResourceOptions options)
{
    uint16_t wkOptions = 0;

    if (options & API::AutomationSessionWebExtensionResourceOptionsPath)
        wkOptions |= _WKAutomationSessionWebExtensionResourceOptionsPath;
    else if (options & API::AutomationSessionWebExtensionResourceOptionsArchivePath)
        wkOptions |= _WKAutomationSessionWebExtensionResourceOptionsArchivePath;
    else
        wkOptions |= _WKAutomationSessionWebExtensionResourceOptionsBase64;

    return static_cast<_WKAutomationSessionWebExtensionResourceOptions>(wkOptions);
}
#endif

void AutomationSessionClient::requestNewPageWithOptions(WebAutomationSession& session, API::AutomationSessionBrowsingContextOptions options, CompletionHandler<void(WebKit::WebPageProxy*)>&& completionHandler)
{
    if (m_delegateMethods.requestNewWebViewWithOptions) {
        [m_delegate.get() _automationSession:wrapper(session) requestNewWebViewWithOptions:toAPI(options) completionHandler:makeBlockPtr([completionHandler = WTFMove(completionHandler)](WKWebView *webView) mutable {
            completionHandler(webView->_page.get());
        }).get()];
    } else
        completionHandler(nullptr);
}

void AutomationSessionClient::requestSwitchToPage(WebAutomationSession& session, WebPageProxy& page, CompletionHandler<void()>&& completionHandler)
{
    if (auto webView = page.cocoaView(); webView && m_delegateMethods.requestSwitchToWebView)
        [m_delegate.get() _automationSession:wrapper(session) requestSwitchToWebView:webView.get() completionHandler:makeBlockPtr(WTFMove(completionHandler)).get()];
    else
        completionHandler();
}

void AutomationSessionClient::requestHideWindowOfPage(WebAutomationSession& session, WebPageProxy& page, CompletionHandler<void()>&& completionHandler)
{
    if (auto webView = page.cocoaView(); webView && m_delegateMethods.requestHideWindowOfWebView)
        [m_delegate.get() _automationSession:wrapper(session) requestHideWindowOfWebView:webView.get() completionHandler:makeBlockPtr(WTFMove(completionHandler)).get()];
    else
        completionHandler();
}

void AutomationSessionClient::requestRestoreWindowOfPage(WebAutomationSession& session, WebPageProxy& page, CompletionHandler<void()>&& completionHandler)
{
    if (auto webView = page.cocoaView(); webView && m_delegateMethods.requestRestoreWindowOfWebView)
        [m_delegate.get() _automationSession:wrapper(session) requestRestoreWindowOfWebView:webView.get() completionHandler:makeBlockPtr(WTFMove(completionHandler)).get()];
    else
        completionHandler();
}

void AutomationSessionClient::requestMaximizeWindowOfPage(WebAutomationSession& session, WebPageProxy& page, CompletionHandler<void()>&& completionHandler)
{
    if (auto webView = page.cocoaView(); webView && m_delegateMethods.requestMaximizeWindowOfWebView)
        [m_delegate.get() _automationSession:wrapper(session) requestMaximizeWindowOfWebView:webView.get() completionHandler:makeBlockPtr(WTFMove(completionHandler)).get()];
    else
        completionHandler();
}

#if ENABLE(WK_WEB_EXTENSIONS_IN_WEBDRIVER)
void AutomationSessionClient::loadWebExtensionWithOptions(WebKit::WebAutomationSession& session, API::AutomationSessionWebExtensionResourceOptions options, const String& resource, CompletionHandler<void(const String&)>&& completionHandler)
{
    if (!m_delegateMethods.loadWebExtensionWithOptions) {
        completionHandler(nullString());
        return;
    }

    [m_delegate.get() _automationSession:wrapper(session) loadWebExtensionWithOptions:toAPI(options) resource:(NSString *)resource completionHandler:makeBlockPtr([completionHandler = WTFMove(completionHandler)](NSString *extensionId) mutable {
        completionHandler(extensionId);
    }).get()];
}

void AutomationSessionClient::unloadWebExtension(WebKit::WebAutomationSession& session, const String& identifier, CompletionHandler<void(bool)>&& completionHandler)
{
    if (!m_delegateMethods.unloadWebExtension) {
        completionHandler(false);
        return;
    }

    [m_delegate.get() _automationSession:wrapper(session) unloadWebExtensionWithIdentifier:identifier completionHandler:makeBlockPtr([completionHandler = WTFMove(completionHandler)](BOOL success) mutable {
        completionHandler(success);
    }).get()];
}
#endif

bool AutomationSessionClient::isShowingJavaScriptDialogOnPage(WebAutomationSession& session, WebPageProxy& page)
{
    if (auto webView = page.cocoaView(); webView && m_delegateMethods.isShowingJavaScriptDialogForWebView)
        return [m_delegate.get() _automationSession:wrapper(session) isShowingJavaScriptDialogForWebView:webView.get()];
    
    return false;
}

void AutomationSessionClient::dismissCurrentJavaScriptDialogOnPage(WebAutomationSession& session, WebPageProxy& page)
{
    if (auto webView = page.cocoaView(); webView && m_delegateMethods.dismissCurrentJavaScriptDialogForWebView)
        [m_delegate.get() _automationSession:wrapper(session) dismissCurrentJavaScriptDialogForWebView:webView.get()];
}

void AutomationSessionClient::acceptCurrentJavaScriptDialogOnPage(WebAutomationSession& session, WebPageProxy& page)
{
    if (auto webView = page.cocoaView(); webView && m_delegateMethods.acceptCurrentJavaScriptDialogForWebView)
        [m_delegate.get() _automationSession:wrapper(session) acceptCurrentJavaScriptDialogForWebView:webView.get()];
}

String AutomationSessionClient::messageOfCurrentJavaScriptDialogOnPage(WebAutomationSession& session, WebPageProxy& page)
{
    if (auto webView = page.cocoaView(); webView && m_delegateMethods.messageOfCurrentJavaScriptDialogForWebView)
        return [m_delegate.get() _automationSession:wrapper(session) messageOfCurrentJavaScriptDialogForWebView:webView.get()];

    return String();
}

void AutomationSessionClient::setUserInputForCurrentJavaScriptPromptOnPage(WebAutomationSession& session, WebPageProxy& page, const String& value)
{
    if (auto webView = page.cocoaView(); webView && m_delegateMethods.setUserInputForCurrentJavaScriptPromptForWebView)
        [m_delegate.get() _automationSession:wrapper(session) setUserInput:value forCurrentJavaScriptDialogForWebView:webView.get()];
}

static std::optional<API::AutomationSessionClient::JavaScriptDialogType> toImpl(_WKAutomationSessionJavaScriptDialogType type)
{
    switch (type) {
    case _WKAutomationSessionJavaScriptDialogTypeNone:
        return std::nullopt;
    case _WKAutomationSessionJavaScriptDialogTypePrompt:
        return API::AutomationSessionClient::JavaScriptDialogType::Prompt;
    case _WKAutomationSessionJavaScriptDialogTypeConfirm:
        return API::AutomationSessionClient::JavaScriptDialogType::Confirm;
    case _WKAutomationSessionJavaScriptDialogTypeAlert:
        return API::AutomationSessionClient::JavaScriptDialogType::Alert;
    }
}

static API::AutomationSessionClient::BrowsingContextPresentation toImpl(_WKAutomationSessionBrowsingContextPresentation presentation)
{
    switch (presentation) {
    case _WKAutomationSessionBrowsingContextPresentationTab:
        return API::AutomationSessionClient::BrowsingContextPresentation::Tab;
    case _WKAutomationSessionBrowsingContextPresentationWindow:
        return API::AutomationSessionClient::BrowsingContextPresentation::Window;
    }
}

std::optional<API::AutomationSessionClient::JavaScriptDialogType> AutomationSessionClient::typeOfCurrentJavaScriptDialogOnPage(WebAutomationSession& session, WebPageProxy& page)
{
    if (auto webView = page.cocoaView(); webView && m_delegateMethods.typeOfCurrentJavaScriptDialogForWebView)
        return toImpl([m_delegate.get() _automationSession:wrapper(session) typeOfCurrentJavaScriptDialogForWebView:webView.get()]);

    return API::AutomationSessionClient::JavaScriptDialogType::Prompt;
}

API::AutomationSessionClient::BrowsingContextPresentation AutomationSessionClient::currentPresentationOfPage(WebAutomationSession& session, WebPageProxy& page)
{
    if (auto webView = page.cocoaView(); webView && m_delegateMethods.currentPresentationForWebView)
        return toImpl([m_delegate.get() _automationSession:wrapper(session) currentPresentationForWebView:webView.get()]);

    return API::AutomationSessionClient::BrowsingContextPresentation::Window;
}

} // namespace WebKit
