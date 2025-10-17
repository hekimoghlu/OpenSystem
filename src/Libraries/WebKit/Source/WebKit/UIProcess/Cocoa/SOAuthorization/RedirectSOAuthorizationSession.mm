/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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
#import "RedirectSOAuthorizationSession.h"

#if HAVE(APP_SSO)

#import "APINavigationAction.h"
#import "WebPageProxy.h"
#import <WebCore/HTTPStatusCodes.h>
#import <WebCore/ResourceResponse.h>
#import <wtf/cocoa/SpanCocoa.h>
#import <wtf/text/MakeString.h>

#define AUTHORIZATIONSESSION_RELEASE_LOG(fmt, ...) RELEASE_LOG(AppSSO, "%p - [InitiatingAction=%s][State=%s] RedirectSOAuthorizationSession::" fmt, this, initiatingActionString().characters(), stateString().characters(), ##__VA_ARGS__)

namespace WebKit {
using namespace WebCore;

Ref<SOAuthorizationSession> RedirectSOAuthorizationSession::create(RetainPtr<WKSOAuthorizationDelegate> delegate, Ref<API::NavigationAction>&& navigationAction, WebPageProxy& page, Callback&& completionHandler)
{
    return adoptRef(*new RedirectSOAuthorizationSession(delegate, WTFMove(navigationAction), page, WTFMove(completionHandler)));
}

RedirectSOAuthorizationSession::RedirectSOAuthorizationSession(RetainPtr<WKSOAuthorizationDelegate> delegate, Ref<API::NavigationAction>&& navigationAction, WebPageProxy& page, Callback&& completionHandler)
    : NavigationSOAuthorizationSession(delegate, WTFMove(navigationAction), page, InitiatingAction::Redirect, WTFMove(completionHandler))
{
}

void RedirectSOAuthorizationSession::fallBackToWebPathInternal()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("fallBackToWebPathInternal: navigationAction=%p", navigationAction());
    invokeCallback(false);
}

void RedirectSOAuthorizationSession::abortInternal()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("abortInternal");
    invokeCallback(true);
}

void RedirectSOAuthorizationSession::completeInternal(const ResourceResponse& response, NSData *data)
{
    AUTHORIZATIONSESSION_RELEASE_LOG("completeInternal: httpState=%d, navigationAction=%p", response.httpStatusCode(), navigationAction());

    RefPtr navigationAction = this->navigationAction();
    ASSERT(navigationAction);
    RefPtr page = this->page();
    // FIXME: Enable the useRedirectionForCurrentNavigation code path for all redirections.
    if ((response.httpStatusCode() != httpStatus302Found && response.httpStatusCode() != httpStatus200OK && !(response.httpStatusCode() == httpStatus307TemporaryRedirect && navigationAction->request().httpMethod() == "POST"_s)) || !page) {
        AUTHORIZATIONSESSION_RELEASE_LOG("completeInternal: httpState=%d page=%d, so falling back to web path.", response.httpStatusCode(), !!page);
        fallBackToWebPathInternal();
        return;
    }

    if (response.httpStatusCode() == httpStatus302Found) {
        invokeCallback(true);
#if PLATFORM(IOS) || PLATFORM(VISION)
        // MobileSafari has a WBSURLSpoofingMitigator, which will not display the provisional URL for navigations without user gestures.
        // For slow loads that are initiated from the MobileSafari Favorites screen, the aforementioned behavior will create a period
        // after authentication completion where the new request to the application site loads with a blank URL and blank page. To
        // work around this issue, we load a page that does a client side redirection to the application site on behalf of the
        // request URL, instead of directly loading a new request. This local page should be super fast to load and therefore will not
        // show an empty URL or a blank page. These changes ensure a relevant URL bar and useful page content during the load.
        if (!navigationAction->isProcessingUserGesture()) {
            page->setShouldSuppressSOAuthorizationInNextNavigationPolicyDecision();
            auto html = makeString("<script>location = '"_s, response.httpHeaderFields().get(HTTPHeaderName::Location), "'</script>"_s).utf8();
            page->loadData(SharedBuffer::create(html.span()), "text/html"_s, "UTF-8"_s, navigationAction->request().url().string(), nullptr, navigationAction->shouldOpenExternalURLsPolicy());
            return;
        }
#endif
        page->loadRequest(ResourceRequest(response.httpHeaderFields().get(HTTPHeaderName::Location)));
        return;
    }
    if (response.httpStatusCode() == httpStatus200OK) {
        invokeCallback(true);
        page->setShouldSuppressSOAuthorizationInNextNavigationPolicyDecision();
        page->loadData(SharedBuffer::create(data), "text/html"_s, "UTF-8"_s, response.url().string(), nullptr, navigationAction->shouldOpenExternalURLsPolicy());
        return;
    }

    ASSERT(response.httpStatusCode() == httpStatus307TemporaryRedirect && navigationAction->request().httpMethod() == "POST"_s);
    page->useRedirectionForCurrentNavigation(response);
    invokeCallback(false);
}

void RedirectSOAuthorizationSession::beforeStart()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("beforeStart");
}

} // namespace WebKit

#undef AUTHORIZATIONSESSION_RELEASE_LOG

#endif
