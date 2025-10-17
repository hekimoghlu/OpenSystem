/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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

#include "AuthenticationResponseJSON.h"
#include "BasicCredential.h"
#include "ExceptionOr.h"
#include "IDLTypes.h"
#include "JSPublicKeyCredentialRequestOptions.h"
#include "RegistrationResponseJSON.h"
#include <wtf/Forward.h>

namespace WebCore {

enum class AuthenticatorAttachment : uint8_t;
class AuthenticatorResponse;
class Document;
typedef IDLRecord<IDLDOMString, IDLBoolean> PublicKeyCredentialClientCapabilities;
typedef std::variant<RegistrationResponseJSON, AuthenticationResponseJSON> PublicKeyCredentialJSON;

struct PublicKeyCredentialCreationOptions;
struct PublicKeyCredentialCreationOptionsJSON;
struct PublicKeyCredentialRequestOptions;
struct PublicKeyCredentialRequestOptionsJSON;
struct AuthenticationExtensionsClientOutputs;

template<typename IDLType> class DOMPromiseDeferred;

class PublicKeyCredential final : public BasicCredential {
public:
    static Ref<PublicKeyCredential> create(Ref<AuthenticatorResponse>&&);

    ArrayBuffer* rawId() const;
    AuthenticatorResponse* response() const { return m_response.ptr(); }
    AuthenticatorAttachment authenticatorAttachment() const;
    AuthenticationExtensionsClientOutputs getClientExtensionResults() const;
    PublicKeyCredentialJSON toJSON();

    static void isUserVerifyingPlatformAuthenticatorAvailable(Document&, DOMPromiseDeferred<IDLBoolean>&&);

    static void getClientCapabilities(Document&, DOMPromiseDeferred<PublicKeyCredentialClientCapabilities>&&);

    static ExceptionOr<PublicKeyCredentialCreationOptions> parseCreationOptionsFromJSON(PublicKeyCredentialCreationOptionsJSON&&);

    static ExceptionOr<PublicKeyCredentialRequestOptions> parseRequestOptionsFromJSON(PublicKeyCredentialRequestOptionsJSON&&);

private:
    PublicKeyCredential(Ref<AuthenticatorResponse>&&);

    Type credentialType() const final { return Type::PublicKey; }

    Ref<AuthenticatorResponse> m_response;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BASIC_CREDENTIAL(PublicKeyCredential, BasicCredential::Type::PublicKey)

#endif // ENABLE(WEB_AUTHN)
