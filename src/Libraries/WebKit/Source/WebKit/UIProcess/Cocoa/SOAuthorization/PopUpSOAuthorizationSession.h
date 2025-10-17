/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#include <wtf/CompletionHandler.h>

OBJC_CLASS WKSOSecretDelegate;
OBJC_CLASS WKWebView;

namespace API {
class NavigationAction;
class PageConfiguration;
}

namespace WebKit {

// FSM: Idle => Active => Completed
class PopUpSOAuthorizationSession final : public SOAuthorizationSession {
public:
    using NewPageCallback = CompletionHandler<void(RefPtr<WebPageProxy>&&)>;
    using UIClientCallback = Function<void(Ref<API::NavigationAction>&&, NewPageCallback&&)>;

    static Ref<SOAuthorizationSession> create(Ref<API::PageConfiguration>&&, RetainPtr<WKSOAuthorizationDelegate>, WebPageProxy&, Ref<API::NavigationAction>&&, NewPageCallback&&, UIClientCallback&&);
    ~PopUpSOAuthorizationSession();

    void close(WKWebView *);

private:
    PopUpSOAuthorizationSession(Ref<API::PageConfiguration>&&, RetainPtr<WKSOAuthorizationDelegate>, WebPageProxy&, Ref<API::NavigationAction>&&, NewPageCallback&&, UIClientCallback&&);

    void shouldStartInternal() final;
    void fallBackToWebPathInternal() final;
    void abortInternal() final;
    void completeInternal(const WebCore::ResourceResponse&, NSData *) final;

    void initSecretWebView();

    Ref<API::PageConfiguration> m_configuration;
    NewPageCallback m_newPageCallback;
    UIClientCallback m_uiClientCallback;

    RetainPtr<WKSOSecretDelegate> m_secretDelegate;
    RetainPtr<WKWebView> m_secretWebView;
};

} // namespace WebKit

#endif
