/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 5, 2025.
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

#include "WebAuthenticationFlags.h"
#include <WebCore/AuthenticatorAssertionResponse.h>
#include <WebCore/AuthenticatorTransport.h>
#include <WebCore/WebAuthenticationConstants.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS ASCAppleIDCredential;
OBJC_CLASS ASCAuthorizationPresentationContext;
OBJC_CLASS ASCAuthorizationPresenter;
OBJC_CLASS ASCLoginChoiceProtocol;
OBJC_CLASS LAContext;
OBJC_CLASS NSError;
OBJC_CLASS WKASCAuthorizationPresenterDelegate;

namespace WebKit {

class AuthenticatorManager;

class AuthenticatorPresenterCoordinator : public RefCountedAndCanMakeWeakPtr<AuthenticatorPresenterCoordinator> {
    WTF_MAKE_TZONE_ALLOCATED(AuthenticatorPresenterCoordinator);
    WTF_MAKE_NONCOPYABLE(AuthenticatorPresenterCoordinator);
public:
    using TransportSet = HashSet<WebCore::AuthenticatorTransport, WTF::IntHash<WebCore::AuthenticatorTransport>, WTF::StrongEnumHashTraits<WebCore::AuthenticatorTransport>>;
    using CredentialRequestHandler = Function<void(ASCAppleIDCredential *, NSError *)>;

    static Ref<AuthenticatorPresenterCoordinator> create(const AuthenticatorManager&, const String& rpId, const TransportSet&, WebCore::ClientDataType, const String& username);
    ~AuthenticatorPresenterCoordinator();

    void updatePresenter(WebAuthenticationStatus);
    void requestPin(uint64_t retries, CompletionHandler<void(const String&)>&&);
    void requestNewPin(uint64_t, CompletionHandler<void(const String&)>&&);
    void selectAssertionResponse(Vector<Ref<WebCore::AuthenticatorAssertionResponse>>&&, WebAuthenticationSource, CompletionHandler<void(WebCore::AuthenticatorAssertionResponse*)>&&);
    void requestLAContextForUserVerification(CompletionHandler<void(LAContext *)>&&);
    void dimissPresenter(WebAuthenticationResult);

    void setCredentialRequestHandler(CredentialRequestHandler&& handler) { m_credentialRequestHandler = WTFMove(handler); }
    void setLAContext(LAContext *);
    void didSelectAssertionResponse(const String& credentialName, LAContext *);
    void setPin(const String&);

private:
    AuthenticatorPresenterCoordinator(const AuthenticatorManager&, const String& rpId, const TransportSet&, WebCore::ClientDataType, const String& username);

    WeakPtr<AuthenticatorManager> m_manager;
    RetainPtr<ASCAuthorizationPresentationContext> m_context;
    RetainPtr<ASCAuthorizationPresenter> m_presenter;
    RetainPtr<WKASCAuthorizationPresenterDelegate> m_presenterDelegate;
    Function<void()> m_delayedPresentation;
#if HAVE(ASC_AUTH_UI)
    bool m_delayedPresentationNeedsSecurityKey { false };
#endif

    CredentialRequestHandler m_credentialRequestHandler;

    CompletionHandler<void(LAContext *)> m_laContextHandler;
    RetainPtr<LAContext> m_laContext;

    CompletionHandler<void(WebCore::AuthenticatorAssertionResponse*)> m_responseHandler;
    HashMap<String, RefPtr<WebCore::AuthenticatorAssertionResponse>> m_credentials;

    CompletionHandler<void(const String&)> m_pinHandler;
#if HAVE(ASC_AUTH_UI)
    bool m_presentedPIN { false };
#endif
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
