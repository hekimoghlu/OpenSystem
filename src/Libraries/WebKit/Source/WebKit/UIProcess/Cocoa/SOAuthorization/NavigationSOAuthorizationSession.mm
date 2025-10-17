/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
#import "NavigationSOAuthorizationSession.h"

#if HAVE(APP_SSO)

#import "Logging.h"
#import "PageLoadState.h"
#import "WebPageProxy.h"
#import <WebCore/ResourceResponse.h>

#define AUTHORIZATIONSESSION_RELEASE_LOG(fmt, ...) RELEASE_LOG(AppSSO, "%p - [InitiatingAction=%s][State=%s] NavigationSOAuthorizationSession::" fmt, this, initiatingActionString().characters(), stateString().characters(), ##__VA_ARGS__)

namespace WebKit {

NavigationSOAuthorizationSession::NavigationSOAuthorizationSession(RetainPtr<WKSOAuthorizationDelegate> delegate, Ref<API::NavigationAction>&& navigationAction, WebPageProxy& page, InitiatingAction action, Callback&& completionHandler)
    : SOAuthorizationSession(delegate, WTFMove(navigationAction), page, action)
    , m_callback(WTFMove(completionHandler))
{
}

NavigationSOAuthorizationSession::~NavigationSOAuthorizationSession()
{
    if (m_callback)
        m_callback(true);
    if (state() == State::Waiting && page())
        page()->removeDidMoveToWindowObserver(*this);
}

void NavigationSOAuthorizationSession::shouldStartInternal()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("shouldStartInternal: m_page=%p", page());

    RefPtr page = this->page();
    ASSERT(page);
    beforeStart();
    if (!page->isInWindow()) {
        AUTHORIZATIONSESSION_RELEASE_LOG("shouldStartInternal: Starting Extensible SSO authentication for a web view that is not attached to a window. Loading will pause until a window is attached.");
        setState(State::Waiting);
        page->addDidMoveToWindowObserver(*this);
        ASSERT(page->mainFrame());
        m_waitingPageActiveURL = page->pageLoadState().activeURL();
        return;
    }
    start();
}

void NavigationSOAuthorizationSession::webViewDidMoveToWindow()
{
    AUTHORIZATIONSESSION_RELEASE_LOG("webViewDidMoveToWindow");
    RefPtr page = this->page();
    if (state() != State::Waiting || !page || !page->isInWindow())
        return;
    if (pageActiveURLDidChangeDuringWaiting()) {
        abort();
        page->removeDidMoveToWindowObserver(*this);
        return;
    }
    start();
    page->removeDidMoveToWindowObserver(*this);
}

bool NavigationSOAuthorizationSession::pageActiveURLDidChangeDuringWaiting() const
{
    AUTHORIZATIONSESSION_RELEASE_LOG("pageActiveURLDidChangeDuringWaiting");
    RefPtr page = this->page();
    return !page || page->pageLoadState().activeURL() != m_waitingPageActiveURL;
}

} // namespace WebKit

#undef AUTHORIZATIONSESSION_RELEASE_LOG

#endif
