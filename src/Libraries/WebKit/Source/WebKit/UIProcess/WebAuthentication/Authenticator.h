/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 8, 2022.
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
#include "WebAuthenticationRequestData.h"
#include <WebCore/AuthenticatorResponse.h>
#include <WebCore/ExceptionData.h>
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>
#include <wtf/spi/cocoa/SecuritySPI.h>

OBJC_CLASS LAContext;

namespace WebCore {
class AuthenticatorAssertionResponse;
}

namespace WebKit {

class Authenticator;
using AuthenticatorObserverRespond = std::variant<Ref<WebCore::AuthenticatorResponse>, WebCore::ExceptionData>;

class AuthenticatorObserver : public AbstractRefCountedAndCanMakeWeakPtr<AuthenticatorObserver> {
public:
    virtual ~AuthenticatorObserver() = default;
    virtual void respondReceived(AuthenticatorObserverRespond&&) = 0;
    virtual void downgrade(Authenticator* id, Ref<Authenticator>&& downgradedAuthenticator) = 0;
    virtual void authenticatorStatusUpdated(WebAuthenticationStatus) = 0;
    virtual void requestPin(uint64_t retries, CompletionHandler<void(const WTF::String&)>&&) = 0;
    virtual void requestNewPin(uint64_t minLength, CompletionHandler<void(const WTF::String&)>&&) = 0;
    virtual void selectAssertionResponse(Vector<Ref<WebCore::AuthenticatorAssertionResponse>>&&, WebAuthenticationSource, CompletionHandler<void(WebCore::AuthenticatorAssertionResponse*)>&&) = 0;
    virtual void decidePolicyForLocalAuthenticator(CompletionHandler<void(LocalAuthenticatorPolicy)>&&) = 0;
    virtual void requestLAContextForUserVerification(CompletionHandler<void(LAContext *)>&&) = 0;
    virtual void cancelRequest() = 0;
};

class Authenticator : public RefCountedAndCanMakeWeakPtr<Authenticator> {
public:
    virtual ~Authenticator() = default;

    void setObserver(AuthenticatorObserver& observer) { m_observer = observer; }

    // This operation is guaranteed to execute asynchronously.
    void handleRequest(const WebAuthenticationRequestData&);

protected:
    Authenticator() = default;

    AuthenticatorObserver* observer() const { return m_observer.get(); }
    const WebAuthenticationRequestData& requestData() const { return m_pendingRequestData; }

    void receiveRespond(AuthenticatorObserverRespond&&) const;

private:
    virtual void makeCredential() = 0;
    virtual void getAssertion() = 0;

    WeakPtr<AuthenticatorObserver> m_observer;
    WebAuthenticationRequestData m_pendingRequestData;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
