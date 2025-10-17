/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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

#include "AuthenticatorResponse.h"
#include "AuthenticatorTransport.h"
#include "RegistrationResponseJSON.h"

namespace WebCore {

class AuthenticatorAttestationResponse : public AuthenticatorResponse {
public:
    static Ref<AuthenticatorAttestationResponse> create(Ref<ArrayBuffer>&& rawId, Ref<ArrayBuffer>&& attestationObject, AuthenticatorAttachment, Vector<AuthenticatorTransport>&&);
    WEBCORE_EXPORT static Ref<AuthenticatorAttestationResponse> create(const Vector<uint8_t>& rawId, const Vector<uint8_t>& attestationObject, AuthenticatorAttachment, Vector<AuthenticatorTransport>&&);

    virtual ~AuthenticatorAttestationResponse() = default;

    ArrayBuffer* attestationObject() const { return m_attestationObject.ptr(); }
    const Vector<AuthenticatorTransport>& getTransports() const { return m_transports; }
    RefPtr<ArrayBuffer> getAuthenticatorData() const;
    RefPtr<ArrayBuffer> getPublicKey() const;
    int64_t getPublicKeyAlgorithm() const;
    RegistrationResponseJSON::AuthenticatorAttestationResponseJSON toJSON();

private:
    AuthenticatorAttestationResponse(Ref<ArrayBuffer>&&, Ref<ArrayBuffer>&&, AuthenticatorAttachment, Vector<AuthenticatorTransport>&&);

    Type type() const final { return Type::Attestation; }
    AuthenticatorResponseData data() const final;

    Ref<ArrayBuffer> m_attestationObject;
    Vector<AuthenticatorTransport> m_transports;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_AUTHENTICATOR_RESPONSE(AuthenticatorAttestationResponse, AuthenticatorResponse::Type::Attestation)

#endif // ENABLE(WEB_AUTHN)
