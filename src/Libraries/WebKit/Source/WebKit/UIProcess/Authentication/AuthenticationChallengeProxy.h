/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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

#include "APIObject.h"
#include "IdentifierTypes.h"
#include <WebCore/AuthenticationChallenge.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class Connection;
}

namespace WebKit {

class AuthenticationDecisionListener;
class WebCredential;
class WebProtectionSpace;

#if HAVE(SEC_KEY_PROXY)
class SecKeyProxyStore;
using WeakPtrSecKeyProxyStore = WeakPtr<SecKeyProxyStore>;
#else
using WeakPtrSecKeyProxyStore = std::nullptr_t;
#endif

class AuthenticationChallengeProxy : public API::ObjectImpl<API::Object::Type::AuthenticationChallenge> {
public:
    static Ref<AuthenticationChallengeProxy> create(WebCore::AuthenticationChallenge&& authenticationChallenge, AuthenticationChallengeIdentifier challengeID, Ref<IPC::Connection>&& connection, WeakPtrSecKeyProxyStore&& secKeyProxyStore)
    {
        return adoptRef(*new AuthenticationChallengeProxy(WTFMove(authenticationChallenge), challengeID, WTFMove(connection), WTFMove(secKeyProxyStore)));
    }

    virtual ~AuthenticationChallengeProxy();

    WebCredential* proposedCredential() const;
    WebProtectionSpace* protectionSpace() const;

    AuthenticationDecisionListener& listener() const { return m_listener.get(); }
    Ref<AuthenticationDecisionListener> protectedListener() const;
    const WebCore::AuthenticationChallenge& core() { return m_coreAuthenticationChallenge; }

private:
    AuthenticationChallengeProxy(WebCore::AuthenticationChallenge&&, AuthenticationChallengeIdentifier, Ref<IPC::Connection>&&, WeakPtrSecKeyProxyStore&&);

#if HAVE(SEC_KEY_PROXY)
    static void sendClientCertificateCredentialOverXpc(IPC::Connection&, SecKeyProxyStore&, AuthenticationChallengeIdentifier, const WebCore::Credential&);
#endif

    WebCore::AuthenticationChallenge m_coreAuthenticationChallenge;
    mutable RefPtr<WebCredential> m_webCredential;
    mutable RefPtr<WebProtectionSpace> m_webProtectionSpace;
    Ref<AuthenticationDecisionListener> m_listener;
};

} // namespace WebKit
