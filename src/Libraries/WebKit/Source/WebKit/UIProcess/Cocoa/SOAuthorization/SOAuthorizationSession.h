/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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

#include <pal/spi/cocoa/AppSSOSPI.h>
#include <wtf/Forward.h>
#include <wtf/RetainPtr.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakObjCPtr.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS SOAuthorization;
OBJC_CLASS WKSOAuthorizationDelegate;

namespace API {
class NavigationAction;
}

namespace WebCore {
class ResourceResponse;
class SecurityOrigin;
}

namespace WebKit {

class WebPageProxy;

enum class SOAuthorizationLoadPolicy : bool;

// A session will only be executed once.
class SOAuthorizationSession : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<SOAuthorizationSession, WTF::DestructionThread::MainRunLoop> {
public:
    enum class InitiatingAction : uint8_t {
        Redirect,
        PopUp,
        SubFrame
    };

    using UICallback = void (^)(BOOL, NSError *);

    virtual ~SOAuthorizationSession();

    // Probably not start immediately.
    void shouldStart();

    // The following should only be called by SOAuthorizationDelegate methods.
    void fallBackToWebPath();
    void abort();
    // Only responses that meet all of the following requirements will be processed:
    // 1) it has the same origin as the request;
    // 2) it has a status code of 302 or 200.
    // Otherwise, it falls back to the web path.
    // Only the following HTTP headers will be processed:
    // { Set-Cookie, Location }.
    void complete(NSHTTPURLResponse *, NSData *);
    void presentViewController(SOAuthorizationViewController, UICallback);

protected:
    // FSM depends on derived classes.
    enum class State : uint8_t {
        Idle,
        Active,
        Waiting,
        Completed
    };

    SOAuthorizationSession(RetainPtr<WKSOAuthorizationDelegate>, Ref<API::NavigationAction>&&, WebPageProxy&, InitiatingAction);

    void start();
    WebPageProxy* page() const { return m_page.get(); }
    State state() const { return m_state; }
    ASCIILiteral stateString() const;
    ASCIILiteral initiatingActionString() const;
    void setState(State state) { m_state = state; }
    const API::NavigationAction* navigationAction() { return m_navigationAction.get(); }
    Ref<API::NavigationAction> releaseNavigationAction();

private:
    virtual void shouldStartInternal() = 0;
    virtual void fallBackToWebPathInternal() = 0;
    virtual void abortInternal() = 0;
    virtual void completeInternal(const WebCore::ResourceResponse&, NSData *) = 0;

    void becomeCompleted();
    void dismissViewController();
#if PLATFORM(MAC)
    void dismissModalSheetIfNecessary();
#endif
    void continueStartAfterGetAuthorizationHints(const String&);
    void continueStartAfterDecidePolicy(const SOAuthorizationLoadPolicy&);

    virtual bool shouldInterruptLoadForCSPFrameAncestorsOrXFrameOptions(const WebCore::ResourceResponse&) { return false; }
    State m_state  { State::Idle };
    RetainPtr<SOAuthorization> m_soAuthorization;
    RefPtr<API::NavigationAction> m_navigationAction;
    WeakPtr<WebPageProxy> m_page;
    InitiatingAction m_action;
    bool m_isInDestructor { false };

    RetainPtr<SOAuthorizationViewController> m_viewController;
#if PLATFORM(MAC)
    RetainPtr<NSWindow> m_sheetWindow;
    RetainPtr<NSObject> m_sheetWindowWillCloseObserver;
    RetainPtr<NSObject> m_presentingWindowDidDeminiaturizeObserver;
    RetainPtr<NSObject> m_applicationDidUnhideObserver;
#endif
};

} // namespace WebKit

#endif
