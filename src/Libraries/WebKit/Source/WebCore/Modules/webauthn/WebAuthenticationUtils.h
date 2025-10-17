/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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

#include "AttestationConveyancePreference.h"
#include "AuthenticatorTransport.h"
#include "BufferSource.h"
#include "CBORValue.h"
#include "SecurityOrigin.h"
#include "WebAuthenticationConstants.h"
#include <wtf/Forward.h>

namespace WebCore {

// Produce a SHA-256 hash of the given RP ID.
WEBCORE_EXPORT Vector<uint8_t> produceRpIdHash(const String& rpId);

WEBCORE_EXPORT Vector<uint8_t> encodeES256PublicKeyAsCBOR(Vector<uint8_t>&& x, Vector<uint8_t>&& y);

// https://www.w3.org/TR/webauthn/#attested-credential-data
WEBCORE_EXPORT Vector<uint8_t> buildAttestedCredentialData(const Vector<uint8_t>& aaguid, const Vector<uint8_t>& credentialId, const Vector<uint8_t>& coseKey);

// https://www.w3.org/TR/webauthn/#sec-authenticator-data
WEBCORE_EXPORT Vector<uint8_t> buildAuthData(const String& rpId, const uint8_t flags, const uint32_t counter, const Vector<uint8_t>& optionalAttestedCredentialData);

WEBCORE_EXPORT cbor::CBORValue::MapValue buildAttestationMap(Vector<uint8_t>&&, String&&, cbor::CBORValue::MapValue&&, const AttestationConveyancePreference&, ShouldZeroAAGUID = ShouldZeroAAGUID::No);

WEBCORE_EXPORT cbor::CBORValue::MapValue buildCredentialDescriptor(const Vector<uint8_t>& credentialId);

// https://www.w3.org/TR/webauthn/#attestation-object
WEBCORE_EXPORT Vector<uint8_t> buildAttestationObject(Vector<uint8_t>&& authData, String&& format, cbor::CBORValue::MapValue&& statementMap, const AttestationConveyancePreference&, ShouldZeroAAGUID = ShouldZeroAAGUID::No);

WEBCORE_EXPORT Ref<ArrayBuffer> buildClientDataJson(ClientDataType /*type*/, const BufferSource& challenge, const SecurityOrigin& /*origin*/, WebAuthn::Scope, const String& topOrigin = { });

WEBCORE_EXPORT Vector<uint8_t> buildClientDataJsonHash(const ArrayBuffer& clientDataJson);

WEBCORE_EXPORT cbor::CBORValue::MapValue buildUserEntityMap(const Vector<uint8_t>& userId, const String& name, const String& displayName);

// encodeRawPublicKey takes X & Y and returns them as a 0x04 || X || Y byte array.
WEBCORE_EXPORT Vector<uint8_t> encodeRawPublicKey(const Vector<uint8_t>& X, const Vector<uint8_t>& Y);

WEBCORE_EXPORT String toString(AuthenticatorTransport);

WEBCORE_EXPORT std::optional<AuthenticatorTransport> convertStringToAuthenticatorTransport(const String& transport);

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
