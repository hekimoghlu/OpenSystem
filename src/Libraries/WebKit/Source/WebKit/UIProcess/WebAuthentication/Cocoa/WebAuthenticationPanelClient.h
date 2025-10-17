/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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

#if ENABLE(WEB_AUTHN)

#import "WKFoundation.h"

#import "APIWebAuthenticationPanelClient.h"
#import <wtf/RetainPtr.h>
#import <wtf/WeakObjCPtr.h>
#import <wtf/WeakPtr.h>

@class _WKWebAuthenticationPanel;
@protocol _WKWebAuthenticationPanelDelegate;

namespace WebKit {

class WebAuthenticationPanelClient final : public API::WebAuthenticationPanelClient, public CanMakeWeakPtr<WebAuthenticationPanelClient> {
public:
    static Ref<WebAuthenticationPanelClient> create(_WKWebAuthenticationPanel *, id<_WKWebAuthenticationPanelDelegate>);

    RetainPtr<id<_WKWebAuthenticationPanelDelegate>> delegate() const;

private:
    WebAuthenticationPanelClient(_WKWebAuthenticationPanel *, id <_WKWebAuthenticationPanelDelegate>);

    // API::WebAuthenticationPanelClient
    void updatePanel(WebAuthenticationStatus) const final;
    void dismissPanel(WebAuthenticationResult) const final;
    void requestPin(uint64_t, CompletionHandler<void(const WTF::String&)>&&) const final;
    void requestNewPin(uint64_t, CompletionHandler<void(const WTF::String&)>&&) const final;
    void selectAssertionResponse(Vector<Ref<WebCore::AuthenticatorAssertionResponse>>&&, WebAuthenticationSource, CompletionHandler<void(WebCore::AuthenticatorAssertionResponse*)>&&) const final;
    void decidePolicyForLocalAuthenticator(CompletionHandler<void(LocalAuthenticatorPolicy)>&&) const final;
    void requestLAContextForUserVerification(CompletionHandler<void(LAContext *)>&&) const final;

    _WKWebAuthenticationPanel *m_panel;
    WeakObjCPtr<id<_WKWebAuthenticationPanelDelegate>> m_delegate;

    struct {
        bool panelUpdateWebAuthenticationPanel : 1;
        bool panelDismissWebAuthenticationPanelWithResult : 1;
        bool panelRequestPinWithRemainingRetriesCompletionHandler : 1;
        bool panelRequestNewPinWithMinLengthCompletionHandler : 1;
        bool panelSelectAssertionResponseSourceCompletionHandler : 1;
        bool panelDecidePolicyForLocalAuthenticatorCompletionHandler : 1;
        bool panelRequestLAContextForUserVerificationCompletionHandler : 1;
    } m_delegateMethods;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
