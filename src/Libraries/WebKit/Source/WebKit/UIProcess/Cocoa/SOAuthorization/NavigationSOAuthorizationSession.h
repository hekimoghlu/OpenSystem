/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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

#if HAVE(APP_SSO)

#include "SOAuthorizationSession.h"
#include "WebViewDidMoveToWindowObserver.h"
#include <wtf/CompletionHandler.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

// When the WebView, owner of the page, is not in the window, the session will then pause
// and later resume after the WebView being moved into the window.
// The reason to apply the above rule to the whole session instead of UI session only is UI
// can be shown out of process, in which case WebKit will not even get notified.
// FSM: Idle => isInWindow => Active => Completed
//      Idle => !isInWindow => Waiting => become isInWindow => Active => Completed
class NavigationSOAuthorizationSession : public SOAuthorizationSession, private WebViewDidMoveToWindowObserver {
public:
    ~NavigationSOAuthorizationSession();

    void ref() const { SOAuthorizationSession::ref(); }
    void deref() const { SOAuthorizationSession::deref(); }

protected:
    using Callback = CompletionHandler<void(bool)>;

    NavigationSOAuthorizationSession(RetainPtr<WKSOAuthorizationDelegate>, Ref<API::NavigationAction>&&, WebPageProxy&, InitiatingAction, Callback&&);

    void invokeCallback(bool intercepted) { m_callback(intercepted); }

private:
    // SOAuthorizationSession
    void shouldStartInternal() final;

    // WebViewDidMoveToWindowObserver
    void webViewDidMoveToWindow() final;

    virtual void beforeStart() = 0;

    bool pageActiveURLDidChangeDuringWaiting() const;

    Callback m_callback;
    String m_waitingPageActiveURL;
};

} // namespace WebKit

#endif
