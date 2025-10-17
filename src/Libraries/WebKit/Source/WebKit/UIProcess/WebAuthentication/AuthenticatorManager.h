/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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

#include "Authenticator.h"
#include "AuthenticatorPresenterCoordinator.h"
#include "AuthenticatorTransportService.h"
#include "WebAuthenticationRequestData.h"
#include <WebCore/AuthenticatorResponse.h>
#include <WebCore/ExceptionData.h>
#include <wtf/CompletionHandler.h>
#include <wtf/HashSet.h>
#include <wtf/Noncopyable.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

OBJC_CLASS LAContext;

namespace API {
class WebAuthenticationPanel;
}

namespace WebKit {

class AuthenticatorManager : public RefCounted<AuthenticatorManager>, public AuthenticatorTransportServiceObserver, public AuthenticatorObserver {
    WTF_MAKE_TZONE_ALLOCATED(AuthenticatorManager);
    WTF_MAKE_NONCOPYABLE(AuthenticatorManager);
public:
    using Respond = std::variant<Ref<WebCore::AuthenticatorResponse>, WebCore::ExceptionData>;
    using Callback = CompletionHandler<void(Respond&&)>;
    using TransportSet = HashSet<WebCore::AuthenticatorTransport, WTF::IntHash<WebCore::AuthenticatorTransport>, WTF::StrongEnumHashTraits<WebCore::AuthenticatorTransport>>;

    USING_CAN_MAKE_WEAKPTR(AuthenticatorTransportServiceObserver);

    const static size_t maxTransportNumber;

    static Ref<AuthenticatorManager> create();
    virtual ~AuthenticatorManager() = default;

    void handleRequest(WebAuthenticationRequestData&&, Callback&&);
    void cancelRequest(const WebCore::PageIdentifier&, const std::optional<WebCore::FrameIdentifier>&); // Called from WebPageProxy/WebProcessProxy.
    void cancelRequest(const API::WebAuthenticationPanel&); // Called from panel clients.
    void cancel(); // Called from the presenter.

    virtual bool isMock() const { return false; }
    virtual bool isVirtual() const { return false; }

    void enableNativeSupport();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

protected:
    AuthenticatorManager();

    RunLoop::Timer& requestTimeOutTimer() { return m_requestTimeOutTimer; }
    void clearStateAsync(); // To void cyclic dependence.
    void clearState();
    void invokePendingCompletionHandler(Respond&&);

    void decidePolicyForLocalAuthenticator(CompletionHandler<void(LocalAuthenticatorPolicy)>&&);
    TransportSet getTransports() const;
    virtual void runPanel();
    void selectAssertionResponse(Vector<Ref<WebCore::AuthenticatorAssertionResponse>>&&, WebAuthenticationSource, CompletionHandler<void(WebCore::AuthenticatorAssertionResponse*)>&&);
    void startDiscovery(const TransportSet&);

private:
    enum class Mode {
        Compatible,
        Native,
    };

    // AuthenticatorTransportServiceObserver
    void authenticatorAdded(Ref<Authenticator>&&) final;
    void serviceStatusUpdated(WebAuthenticationStatus) final;

    // AuthenticatorObserver
    void respondReceived(Respond&&) final;
    void downgrade(Authenticator* id, Ref<Authenticator>&& downgradedAuthenticator) final;
    void authenticatorStatusUpdated(WebAuthenticationStatus) final;
    void requestPin(uint64_t retries, CompletionHandler<void(const WTF::String&)>&&) final;
    void requestNewPin(uint64_t minLength, CompletionHandler<void(const WTF::String&)>&&) final;
    void requestLAContextForUserVerification(CompletionHandler<void(LAContext *)>&&) final;
    void cancelRequest() final;

    // Overriden by MockAuthenticatorManager.
    virtual Ref<AuthenticatorTransportService> createService(WebCore::AuthenticatorTransport, AuthenticatorTransportServiceObserver&) const;
    // Overriden to return every exception for tests to confirm.
    virtual void respondReceivedInternal(Respond&&) { }
    virtual void filterTransports(TransportSet&) const;
    virtual void runPresenterInternal(const TransportSet&);

    void initTimeOutTimer();
    void timeOutTimerFired();
    void runPresenter();
    void restartDiscovery();
    void dispatchPanelClientCall(Function<void(const API::WebAuthenticationPanel&)>&&) const;

    // Request: We only allow one request per time. A new request will cancel any pending ones.
    WebAuthenticationRequestData m_pendingRequestData;
    Callback m_pendingCompletionHandler; // Should not be invoked directly, use invokePendingCompletionHandler.
    RunLoop::Timer m_requestTimeOutTimer;
    RefPtr<AuthenticatorPresenterCoordinator> m_presenter;

    Vector<Ref<AuthenticatorTransportService>> m_services;
    HashSet<Ref<Authenticator>> m_authenticators;

    Mode m_mode { Mode::Compatible };
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
