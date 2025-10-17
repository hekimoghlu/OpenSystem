/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#include "CryptoAlgorithmX25519.h"

#include "CryptoAlgorithmEcKeyParams.h"
#include "CryptoAlgorithmX25519Params.h"
#include "CryptoKeyOKP.h"
#include "ScriptExecutionContext.h"
#include <wtf/CryptographicUtilities.h>

namespace WebCore {

Ref<CryptoAlgorithm> CryptoAlgorithmX25519::create()
{
    return adoptRef(*new CryptoAlgorithmX25519);
}

CryptoAlgorithmIdentifier CryptoAlgorithmX25519::identifier() const
{
    return s_identifier;
}

void CryptoAlgorithmX25519::generateKey(const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap usages, KeyOrKeyPairCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&)
{
    if (usages & (CryptoKeyUsageEncrypt | CryptoKeyUsageDecrypt | CryptoKeyUsageSign | CryptoKeyUsageVerify | CryptoKeyUsageWrapKey | CryptoKeyUsageUnwrapKey)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }

    auto result = CryptoKeyOKP::generatePair(CryptoAlgorithmIdentifier::X25519, CryptoKeyOKP::NamedCurve::X25519, extractable, usages);
    if (result.hasException()) {
        exceptionCallback(result.releaseException().code());
        return;
    }

    auto pair = result.releaseReturnValue();
    pair.publicKey->setUsagesBitmap(0);
    pair.privateKey->setUsagesBitmap(pair.privateKey->usagesBitmap() & (CryptoKeyUsageDeriveKey | CryptoKeyUsageDeriveBits));
    callback(WTFMove(pair));
}

#if !PLATFORM(COCOA) && !USE(GCRYPT)
std::optional<Vector<uint8_t>> CryptoAlgorithmX25519::platformDeriveBits(const CryptoKeyOKP&, const CryptoKeyOKP&)
{
    return std::nullopt;
}
#endif

void CryptoAlgorithmX25519::deriveBits(const CryptoAlgorithmParameters& parameters, Ref<CryptoKey>&& baseKey, std::optional<size_t> length, VectorCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    if (baseKey->type() != CryptoKey::Type::Private) {
        exceptionCallback(ExceptionCode::InvalidAccessError);
        return;
    }
    auto& ecParameters = downcast<CryptoAlgorithmX25519Params>(parameters);
    ASSERT(ecParameters.publicKey);
    if (ecParameters.publicKey->type() != CryptoKey::Type::Public) {
        exceptionCallback(ExceptionCode::InvalidAccessError);
        return;
    }
    if (baseKey->algorithmIdentifier() != ecParameters.publicKey->algorithmIdentifier()) {
        exceptionCallback(ExceptionCode::InvalidAccessError);
        return;
    }
    auto& ecBaseKey = downcast<CryptoKeyOKP>(baseKey.get());
    auto& ecPublicKey = downcast<CryptoKeyOKP>(*(ecParameters.publicKey.get()));
    if (ecBaseKey.namedCurve() != ecPublicKey.namedCurve()) {
        exceptionCallback(ExceptionCode::InvalidAccessError);
        return;
    }

    // Return an empty string doesn't make much sense, but truncating either at all.
    // https://github.com/WICG/webcrypto-secure-curves/pull/29
    if (length && !(*length)) {
        // Avoid executing the key-derivation, since we are going to return an empty string.
        callback({ });
        return;
    }

    auto unifiedCallback = [callback = WTFMove(callback), exceptionCallback = WTFMove(exceptionCallback)](std::optional<Vector<uint8_t>>&& derivedKey, std::optional<size_t> length) mutable {
        if (!derivedKey) {
            exceptionCallback(ExceptionCode::OperationError);
            return;
        }
        if (!length) {
            callback(WTFMove(*derivedKey));
            return;
        }
#if !HAVE(X25519_ZERO_CHECKS)
        // https://datatracker.ietf.org/doc/html/rfc7748#section-6.1
        constexpr auto expectedOutputSize = 32;
        constexpr std::array<uint8_t, expectedOutputSize> zeros { };
        if (derivedKey->size() != expectedOutputSize || !constantTimeMemcmp(derivedKey->span(), zeros)) {
            exceptionCallback(ExceptionCode::OperationError);
            return;
        }
#endif
        auto lengthInBytes = std::ceil(*length / 8.);
        if (lengthInBytes > (*derivedKey).size()) {
            exceptionCallback(ExceptionCode::OperationError);
            return;
        }
        (*derivedKey).shrink(lengthInBytes);
        callback(WTFMove(*derivedKey));
    };
    // This is a special case that can't use dispatchOperation() because it bundles
    // the result validation and callback dispatch into unifiedCallback.
    workQueue.dispatch(
        [baseKey = WTFMove(baseKey), publicKey = ecParameters.publicKey, length, unifiedCallback = WTFMove(unifiedCallback), contextIdentifier = context.identifier()]() mutable {
            auto derivedKey = platformDeriveBits(downcast<CryptoKeyOKP>(baseKey.get()), downcast<CryptoKeyOKP>(*publicKey));
            ScriptExecutionContext::postTaskTo(contextIdentifier, [derivedKey = WTFMove(derivedKey), length, unifiedCallback = WTFMove(unifiedCallback)](auto&) mutable {
                unifiedCallback(WTFMove(derivedKey), length);
            });
        });
}

void CryptoAlgorithmX25519::importKey(CryptoKeyFormat format, KeyData&& data, const CryptoAlgorithmParameters&, bool extractable, CryptoKeyUsageBitmap usages, KeyCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    RefPtr<CryptoKeyOKP> result;
    switch (format) {
    case CryptoKeyFormat::Jwk: {
        JsonWebKey key = WTFMove(std::get<JsonWebKey>(data));

        bool isUsagesAllowed = false;
        if (!key.d.isNull()) {
            isUsagesAllowed = isUsagesAllowed || !(usages ^ CryptoKeyUsageDeriveKey);
            isUsagesAllowed = isUsagesAllowed || !(usages ^ CryptoKeyUsageDeriveBits);
            isUsagesAllowed = isUsagesAllowed || !(usages ^ (CryptoKeyUsageDeriveKey | CryptoKeyUsageDeriveBits));
        }
        isUsagesAllowed = isUsagesAllowed || !usages;
        if (!isUsagesAllowed) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }

        if (usages && !key.use.isNull() && key.use != "enc"_s) {
            exceptionCallback(ExceptionCode::DataError);
            return;
        }

        result = CryptoKeyOKP::importJwk(CryptoAlgorithmIdentifier::X25519, CryptoKeyOKP::NamedCurve::X25519, WTFMove(key), extractable, usages);
        break;
    }
    case CryptoKeyFormat::Raw:
        if (usages) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }
        result = CryptoKeyOKP::importRaw(CryptoAlgorithmIdentifier::X25519, CryptoKeyOKP::NamedCurve::X25519, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    case CryptoKeyFormat::Spki:
        if (usages) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }
        result = CryptoKeyOKP::importSpki(CryptoAlgorithmIdentifier::X25519, CryptoKeyOKP::NamedCurve::X25519, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    case CryptoKeyFormat::Pkcs8:
        if (usages && (usages ^ CryptoKeyUsageDeriveKey) && (usages ^ CryptoKeyUsageDeriveBits) && (usages ^ (CryptoKeyUsageDeriveKey | CryptoKeyUsageDeriveBits))) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }
        result = CryptoKeyOKP::importPkcs8(CryptoAlgorithmIdentifier::X25519, CryptoKeyOKP::NamedCurve::X25519, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    }
    if (!result) {
        exceptionCallback(ExceptionCode::DataError);
        return;
    }

    callback(*result);
}

void CryptoAlgorithmX25519::exportKey(CryptoKeyFormat format, Ref<CryptoKey>&& key, KeyDataCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    const auto& ecKey = downcast<CryptoKeyOKP>(key.get());

    if (!ecKey.keySizeInBits()) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    KeyData result;
    switch (format) {
    case CryptoKeyFormat::Jwk: {
        auto jwk = ecKey.exportJwk();
        if (jwk.hasException()) {
            exceptionCallback(jwk.releaseException().code());
            return;
        }
        result = jwk.releaseReturnValue();
        break;
    }
    case CryptoKeyFormat::Raw: {
        auto raw = ecKey.exportRaw();
        if (raw.hasException()) {
            exceptionCallback(raw.releaseException().code());
            return;
        }
        result = raw.releaseReturnValue();
        break;
    }
    case CryptoKeyFormat::Spki: {
        auto spki = ecKey.exportSpki();
        if (spki.hasException()) {
            exceptionCallback(spki.releaseException().code());
            return;
        }
        result = spki.releaseReturnValue();
        break;
    }
    case CryptoKeyFormat::Pkcs8: {
        auto pkcs8 = ecKey.exportPkcs8();
        if (pkcs8.hasException()) {
            exceptionCallback(pkcs8.releaseException().code());
            return;
        }
        result = pkcs8.releaseReturnValue();
        break;
    }
    }

    callback(format, WTFMove(result));
}

} // namespace WebCore
