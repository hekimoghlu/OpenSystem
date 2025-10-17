/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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
#import "PopUpSOAuthorizationSession.h"

#if HAVE(APP_SSO)

#import "APINavigationAction.h"
#import "WKWebViewConfigurationInternal.h"
#import "WKWebViewInternal.h"
#import "WebPageProxy.h"
#import <WebKit/WKNavigationDelegatePrivate.h>
#import <WebKit/WKPreferencesPrivate.h>
#import <WebKit/WKUIDelegate.h>
#import <WebKit/WKWebViewConfigurationPrivate.h>
#import <WebCore/HTTPStatusCodes.h>
#import <WebCore/ResourceResponse.h>
#import <wtf/BlockPtr.h>

@interface WKSOSecretDelegate : NSObject <WKNavigationDelegate, WKUIDelegate> {
@private
    ThreadSafeWeakPtr<WebKit::PopUpSOAuthorizationSession> _weakSession;
    BOOL _isFirstNavigation;
}

- (instancetype)initWithSession:(WebKit::PopUpSOAuthorizationSession&)session;

@end

@implementation WKSOSecretDelegate

- (instancetype)initWithSession:(WebKit::PopUpSOAuthorizationSession&)session
{
    if ((self = [super init])) {
        _weakSession = session;
        _isFirstNavigation = YES;
    }
    return self;
}

// WKUIDelegate
- (void)webViewDidClose:(WKWebView *)webView
{
    auto strongSession = _weakSession.get();
    if (!strongSession)
        return;
    strongSession->close(webView);
}

// WKNavigationDelegate
- (void)webView:(WKWebView *)webView decidePolicyForNavigationAction:(WKNavigationAction *)navigationAction decisionHandler:(void (^)(WKNavigationActionPolicy))decisionHandler
{
    // FIXME<rdar://problem/48787839>: We should restrict the load to only substitute data.
    // Use the following heuristic as a workaround right now.
    // Ignore the first load in the secret window, which navigates to the authentication URL.
    if (_isFirstNavigation) {
        _isFirstNavigation = NO;
        decisionHandler(WKNavigationActionPolicyCancel);
        return;
    }
    decisionHandler(_WKNavigationActionPolicyAllowWithoutTryingAppLink);
}

- (void)webView:(WKWebView *)webView didFinishNavigation:(WKNavigation *)navigation
{
    auto strongSession = _weakSession.get();
    if (!strongSession)
        return;
    strongSession->close(webView);
}

@end

#define AUTHORIZATIONSESSION_RELEASE_LOG(fmt, ...) RELEASE_LOG(AppSSO, "%p - [InitiatingAction=%s][State=%s] PopUpSOAuthorizationSession::" fmt, this, initiatingActionString().characters(), stateString().characters(), ##__VA_ARGS__)

namespace WebKit {

Ref<SOAuthorizationSession> PopUpSOAuthorizationSession::create(Ref<API::PageConfiguration>&& configuration, RetainPtr<WKSOAuthorizationDelegate> delegate, WebPageProxy& page, Ref<API::NavigationAction>&& navigationAction, NewPageCallback&& newPageCallback, UIClientCallback&& uiClientCallback)
{
    return adoptRef(*new PopUpSOAuthorizationSession(WTFMove(configuration), delegate, page, WTFMove(navigationAction), WTFMove(newPageCallback), WTFMove(uiClientCallback)));
}

PopUpSOAuthorizationSession::PopUpSOAuthorizationSession(Ref<API::PageConfiguration>&& configuration, RetainPtr<WKSOAuthorizationDelegate> delegate, WebPageProxy& page, Ref<API::NavigationAction>&& navigationAction, NewPageCallback&& newPageCallback, UIClientCallback&& uiClientCallback)
    : SOAuthorizationSession(delegate, WTFMove(navigationAction), page, InitiatingAction::PopUp)
    , m_configuration(WTFMove(configuration))
    , m_newPageCallback(WTFMove(newPageCallback))
    , m_uiClientCallback(WTFMove(uiClientCallback))
{
}

PopUpSOAuthorizationSession::~PopUpSOAuthorizationSession()
{
    ASSERT(state() != State::Waiting);
    if (m_newPageCallback)
        m_newPageCallback(nullptr);
}

void PopUpSOAuthorizationSession::shouldStartInternal()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("shouldStartInternal: m_page=%p", page());
    ASSERT(page() && page()->isInWindow());
    start();
}

void PopUpSOAuthorizationSession::fallBackToWebPathInternal()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("fallBackToWebPathInternal");
    m_uiClientCallback(releaseNavigationAction(), WTFMove(m_newPageCallback));
}

void PopUpSOAuthorizationSession::abortInternal()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("abortInternal: m_page=%p", page());
    if (!page()) {
        m_newPageCallback(nullptr);
        return;
    }

    initSecretWebView();
    if (!m_secretWebView) {
        m_newPageCallback(nullptr);
        return;
    }

    m_newPageCallback(m_secretWebView->_page.get());
    [m_secretWebView evaluateJavaScript: @"window.close()" completionHandler:nil];
}

void PopUpSOAuthorizationSession::completeInternal(const WebCore::ResourceResponse& response, NSData *data)
{
    AUTHORIZATIONSESSION_RELEASE_LOG("completeInternal: httpState=%d", response.httpStatusCode());
    if (response.httpStatusCode() != httpStatus200OK || !page()) {
        fallBackToWebPathInternal();
        return;
    }

    initSecretWebView();
    if (!m_secretWebView) {
        fallBackToWebPathInternal();
        return;
    }

    m_newPageCallback(m_secretWebView->_page.get());
    [m_secretWebView loadData:data MIMEType:@"text/html" characterEncodingName:@"UTF-8" baseURL:response.url()];
}

void PopUpSOAuthorizationSession::close(WKWebView *webView)
{
    AUTHORIZATIONSESSION_RELEASE_LOG("close");
    if (!m_secretWebView)
        return;
    if (state() != State::Completed || webView != m_secretWebView.get()) {
        ASSERT_NOT_REACHED();
        return;
    }
    m_secretWebView = nullptr;
    WTFLogAlways("SecretWebView is cleaned.");
}

void PopUpSOAuthorizationSession::initSecretWebView()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("initSecretWebView");
    ASSERT(page());
    RetainPtr configuration = wrapper(m_configuration);
    auto secretViewPreferences = adoptNS([[configuration preferences] copy]);
    [secretViewPreferences _setExtensibleSSOEnabled:NO];
    [configuration setPreferences:secretViewPreferences.get()];
    m_secretWebView = adoptNS([[WKWebView alloc] initWithFrame:CGRectZero configuration:configuration.get()]);

    m_secretDelegate = adoptNS([[WKSOSecretDelegate alloc] initWithSession:*this]);
    [m_secretWebView setUIDelegate:m_secretDelegate.get()];
    [m_secretWebView setNavigationDelegate:m_secretDelegate.get()];

    RELEASE_ASSERT(!m_secretWebView->_page->preferences().isExtensibleSSOEnabled());
    WTFLogAlways("SecretWebView is created.");
}

} // namespace WebKit

#undef AUTHORIZATIONSESSION_RELEASE_LOG

#endif
