/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
#include "LocalConnection.h"
#include <wtf/UniqueRef.h>

OBJC_CLASS LAContext;

namespace WebCore {
class AuthenticatorAttestationResponse;
class AuthenticatorAssertionResponse;
}

namespace WebKit {

BOOL shouldUseAlternateKeychainAttribute();

class LocalAuthenticator final : public Authenticator {
public:
    // Here is the FSM.
    // MakeCredential: Init => RequestReceived => PolicyDecided => UserVerified => (Attested) => End
    // GetAssertion: Init => RequestReceived => ResponseSelected => UserVerified => End
    enum class State {
        Init,
        RequestReceived,
        UserVerified,
        Attested,
        ResponseSelected,
        PolicyDecided,
    };

    static Ref<LocalAuthenticator> create(Ref<LocalConnection>&& connection)
    {
        return adoptRef(*new LocalAuthenticator(WTFMove(connection)));
    }

    static void clearAllCredentials();

private:
    explicit LocalAuthenticator(Ref<LocalConnection>&&);

    std::optional<WebCore::ExceptionData> processClientExtensions(std::variant<Ref<WebCore::AuthenticatorAttestationResponse>, Ref<WebCore::AuthenticatorAssertionResponse>>);

    void makeCredential() final;
    void continueMakeCredentialAfterReceivingLAContext(LAContext *);
    void continueMakeCredentialAfterUserVerification(SecAccessControlRef, LocalConnection::UserVerification, LAContext *);
    void finishMakeCredential(Vector<uint8_t>&& credentialId, Vector<uint8_t>&& attestationObject, std::optional<WebCore::ExceptionData>);

    void getAssertion() final;
    void continueGetAssertionAfterResponseSelected(Ref<WebCore::AuthenticatorAssertionResponse>&&);
    void continueGetAssertionAfterUserVerification(Ref<WebCore::AuthenticatorAssertionResponse>&&, LocalConnection::UserVerification, LAContext *);

    void receiveException(WebCore::ExceptionData&&, WebAuthenticationStatus = WebAuthenticationStatus::LAError) const;
    void deleteDuplicateCredential() const;
    bool validateUserVerification(LocalConnection::UserVerification) const;

    std::optional<WebCore::ExceptionData> processLargeBlobExtension(const WebCore::PublicKeyCredentialCreationOptions&, WebCore::AuthenticationExtensionsClientOutputs& extensionOutputs);
    std::optional<WebCore::ExceptionData> processLargeBlobExtension(const WebCore::PublicKeyCredentialRequestOptions&, WebCore::AuthenticationExtensionsClientOutputs& extensionOutputs, const Ref<WebCore::AuthenticatorAssertionResponse>&);

    std::optional<Vector<Ref<WebCore::AuthenticatorAssertionResponse>>> getExistingCredentials(const String& rpId);

    Ref<LocalConnection> protectedConnection() const { return m_connection; }

    State m_state { State::Init };
    Ref<LocalConnection> m_connection;
    Vector<Ref<WebCore::AuthenticatorAssertionResponse>> m_existingCredentials;
    RetainPtr<NSData> m_provisionalCredentialId;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
