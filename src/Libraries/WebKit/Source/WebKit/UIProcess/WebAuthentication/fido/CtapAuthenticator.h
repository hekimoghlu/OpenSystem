/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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

#include "FidoAuthenticator.h"
#include <WebCore/AuthenticatorGetInfoResponse.h>
#include <WebCore/CryptoKeyEC.h>

namespace fido {
namespace pin {
class TokenRequest;
}
}

namespace WebKit {

class CtapDriver;

class CtapAuthenticator final : public FidoAuthenticator {
public:
    static Ref<CtapAuthenticator> create(Ref<CtapDriver>&& driver, fido::AuthenticatorGetInfoResponse&& info)
    {
        return adoptRef(*new CtapAuthenticator(WTFMove(driver), WTFMove(info)));
    }

private:
    explicit CtapAuthenticator(Ref<CtapDriver>&&, fido::AuthenticatorGetInfoResponse&&);

    void makeCredential() final;
    void continueMakeCredentialAfterResponseReceived(Vector<uint8_t>&&);
    void getAssertion() final;
    void continueGetAssertionAfterResponseReceived(Vector<uint8_t>&&);
    void continueGetNextAssertionAfterResponseReceived(Vector<uint8_t>&&);

    void getRetries();
    void continueGetKeyAgreementAfterGetRetries(Vector<uint8_t>&&);
    void continueRequestPinAfterGetKeyAgreement(Vector<uint8_t>&&, uint64_t retries);
    void continueGetPinTokenAfterRequestPin(const String& pin, const WebCore::CryptoKeyEC&);
    void continueRequestAfterGetPinToken(Vector<uint8_t>&&, const fido::pin::TokenRequest&);
    bool tryRestartPin(const fido::CtapDeviceResponseCode&);

    bool tryDowngrade();

    Vector<WebCore::AuthenticatorTransport> transports() const;

    String aaguidForDebugging() const;

    bool isUVSetup() const;

    void continueSetupPinAfterCommand(Vector<uint8_t>&&, const String& pin, Ref<WebCore::CryptoKeyEC> peerKey);
    void continueSetupPinAfterGetKeyAgreement(Vector<uint8_t>&&, const String& pin);
    void performAuthenticatorSelectionForSetupPin();
    void setupPin();

    fido::AuthenticatorGetInfoResponse m_info;
    bool m_isDowngraded { false };
    bool m_isKeyStoreFull { false };
    size_t m_remainingAssertionResponses { 0 };
    Vector<Ref<WebCore::AuthenticatorAssertionResponse>> m_assertionResponses;
    Vector<uint8_t> m_pinAuth;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
