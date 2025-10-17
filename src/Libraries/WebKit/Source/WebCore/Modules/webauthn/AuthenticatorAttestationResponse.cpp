/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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
#include "config.h"
#include "AuthenticatorAttestationResponse.h"

#if ENABLE(WEB_AUTHN)

#include "AuthenticatorResponseData.h"
#include "CBORReader.h"
#include "CryptoAlgorithmECDH.h"
#include "CryptoKeyEC.h"
#include "WebAuthenticationUtils.h"
#include <wtf/text/Base64.h>

namespace WebCore {

static std::optional<cbor::CBORValue> coseKeyForAttestationObject(Ref<ArrayBuffer> attObj)
{
    auto decodedResponse = cbor::CBORReader::read(attObj->toVector());
    if (!decodedResponse || !decodedResponse->isMap()) {
        ASSERT_NOT_REACHED();
        return std::nullopt;
    }
    const auto& attObjMap = decodedResponse->getMap();
    auto it = attObjMap.find(cbor::CBORValue("authData"));
    if (it == attObjMap.end() || !it->second.isByteString()) {
        ASSERT_NOT_REACHED();
        return std::nullopt;
    }
    auto authData = it->second.getByteString();
    const size_t credentialIdLengthOffset = rpIdHashLength + flagsLength + signCounterLength + aaguidLength;
    if (authData.size() < credentialIdLengthOffset + credentialIdLengthLength)
        return std::nullopt;

    const size_t credentialIdLength = (static_cast<size_t>(authData[credentialIdLengthOffset]) << 8) | static_cast<size_t>(authData[credentialIdLengthOffset + 1]);
    const size_t cosePublicKeyOffset = credentialIdLengthOffset + credentialIdLengthLength + credentialIdLength;
    if (authData.size() <= cosePublicKeyOffset)
        return std::nullopt;

    const size_t cosePublicKeyLength = authData.size() - cosePublicKeyOffset;
    Vector<uint8_t> cosePublicKey(authData.subspan(cosePublicKeyOffset, cosePublicKeyLength));
    return cbor::CBORReader::read(cosePublicKey);
}

Ref<AuthenticatorAttestationResponse> AuthenticatorAttestationResponse::create(Ref<ArrayBuffer>&& rawId, Ref<ArrayBuffer>&& attestationObject, AuthenticatorAttachment attachment, Vector<AuthenticatorTransport>&& transports)
{
    return adoptRef(*new AuthenticatorAttestationResponse(WTFMove(rawId), WTFMove(attestationObject), attachment, WTFMove(transports)));
}

Ref<AuthenticatorAttestationResponse> AuthenticatorAttestationResponse::create(const Vector<uint8_t>& rawId, const Vector<uint8_t>& attestationObject, AuthenticatorAttachment attachment, Vector<AuthenticatorTransport>&& transports)
{
    return create(ArrayBuffer::create(rawId), ArrayBuffer::create(attestationObject), attachment, WTFMove(transports));
}

AuthenticatorAttestationResponse::AuthenticatorAttestationResponse(Ref<ArrayBuffer>&& rawId, Ref<ArrayBuffer>&& attestationObject, AuthenticatorAttachment attachment, Vector<AuthenticatorTransport>&& transports)
    : AuthenticatorResponse(WTFMove(rawId), attachment)
    , m_attestationObject(WTFMove(attestationObject))
    , m_transports(WTFMove(transports))
{
}

AuthenticatorResponseData AuthenticatorAttestationResponse::data() const
{
    auto data = AuthenticatorResponse::data();
    data.isAuthenticatorAttestationResponse = true;
    data.attestationObject = m_attestationObject.copyRef();
    data.transports = m_transports;
    return data;
}

RefPtr<ArrayBuffer> AuthenticatorAttestationResponse::getAuthenticatorData() const
{
    auto decodedResponse = cbor::CBORReader::read(m_attestationObject->toVector());
    if (!decodedResponse || !decodedResponse->isMap()) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }
    const auto& attObjMap = decodedResponse->getMap();
    auto it = attObjMap.find(cbor::CBORValue("authData"));
    if (it == attObjMap.end() || !it->second.isByteString()) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }
    auto authData = it->second.getByteString();
    return ArrayBuffer::tryCreate(authData);
}

int64_t AuthenticatorAttestationResponse::getPublicKeyAlgorithm() const
{
    auto key = coseKeyForAttestationObject(m_attestationObject);
    if (!key || !key->isMap())
        return 0;
    auto& keyMap = key->getMap();

    auto it = keyMap.find(cbor::CBORValue(COSE::alg));
    if (it == keyMap.end() || !it->second.isInteger()) {
        ASSERT_NOT_REACHED();
        return 0;
    }
    return it->second.getInteger();
}

RefPtr<ArrayBuffer> AuthenticatorAttestationResponse::getPublicKey() const
{
    auto key = coseKeyForAttestationObject(m_attestationObject);
    if (!key || !key->isMap())
        return nullptr;
    auto& keyMap = key->getMap();

    auto it = keyMap.find(cbor::CBORValue(COSE::alg));
    if (it == keyMap.end() || !it->second.isInteger()) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }
    auto alg = it->second.getInteger();

    it = keyMap.find(cbor::CBORValue(COSE::kty));
    if (it == keyMap.end() || !it->second.isInteger()) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }
    auto kty = it->second.getInteger();

    std::optional<int64_t> crv;
    it = keyMap.find(cbor::CBORValue(COSE::crv));
    if (it != keyMap.end() && it->second.isInteger())
        crv = it->second.getInteger();

    switch (alg) {
    case COSE::ES256: {
        if (kty != COSE::EC2 || crv != COSE::P_256)
            return nullptr;

        auto it = keyMap.find(cbor::CBORValue(COSE::x));
        if (it == keyMap.end() || !it->second.isByteString()) {
            ASSERT_NOT_REACHED();
            return nullptr;
        }
        auto x = it->second.getByteString();

        it = keyMap.find(cbor::CBORValue(COSE::y));
        if (it == keyMap.end() || !it->second.isByteString()) {
            ASSERT_NOT_REACHED();
            return nullptr;
        }
        auto y = it->second.getByteString();
        auto peerKey = CryptoKeyEC::importRaw(CryptoAlgorithmIdentifier::ECDH, "P-256"_s, encodeRawPublicKey(x, y), true, CryptoKeyUsageDeriveBits);

        if (!peerKey)
            return nullptr;
        auto keySpki = peerKey->exportSpki().releaseReturnValue();
        return ArrayBuffer::tryCreate(keySpki);
    }
    default:
        break;
    }

    return nullptr;
}

RegistrationResponseJSON::AuthenticatorAttestationResponseJSON AuthenticatorAttestationResponse::toJSON()
{
    Vector<String> transports;
    for (auto transport : getTransports())
        transports.append(toString(transport));
    RegistrationResponseJSON::AuthenticatorAttestationResponseJSON value;
    if (auto clientData = clientDataJSON())
        value.clientDataJSON = base64URLEncodeToString(clientData->span());
    value.transports = transports;
    if (auto authData = getAuthenticatorData())
        value.authenticatorData = base64URLEncodeToString(authData->span());
    if (auto publicKey = getPublicKey())
        value.publicKey = base64URLEncodeToString(publicKey->span());
    if (auto attestationObj = attestationObject())
        value.attestationObject = base64URLEncodeToString(attestationObj->span());
    value.publicKeyAlgorithm = getPublicKeyAlgorithm();

    return value;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
