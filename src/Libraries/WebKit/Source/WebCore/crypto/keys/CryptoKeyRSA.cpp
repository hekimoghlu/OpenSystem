/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 24, 2024.
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
#include "CryptoKeyRSA.h"

#include "CryptoKeyRSAComponents.h"
#include "JsonWebKey.h"
#include <wtf/text/Base64.h>

namespace WebCore {

RefPtr<CryptoKeyRSA> CryptoKeyRSA::importJwk(CryptoAlgorithmIdentifier algorithm, std::optional<CryptoAlgorithmIdentifier> hash, JsonWebKey&& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    if (keyData.kty != "RSA"_s)
        return nullptr;
    if (keyData.key_ops && ((keyData.usages & usages) != usages))
        return nullptr;
    if (keyData.ext && !keyData.ext.value() && extractable)
        return nullptr;

    if (keyData.n.isNull() || keyData.e.isNull())
        return nullptr;
    auto modulus = base64URLDecode(keyData.n);
    if (!modulus)
        return nullptr;
    // Per RFC 7518 Section 6.3.1.1: https://tools.ietf.org/html/rfc7518#section-6.3.1.1
    if (!modulus->isEmpty() && !modulus->at(0))
        modulus->remove(0);
    auto exponent = base64URLDecode(keyData.e);
    if (!exponent)
        return nullptr;
    if (keyData.d.isNull()) {
        // import public key
        auto publicKeyComponents = CryptoKeyRSAComponents::createPublic(WTFMove(*modulus), WTFMove(*exponent));
        // Notice: CryptoAlgorithmIdentifier::SHA_1 is just a placeholder. It should not have any effect if hash is std::nullopt.
        return CryptoKeyRSA::create(algorithm, hash.value_or(CryptoAlgorithmIdentifier::SHA_1), !!hash, *publicKeyComponents, extractable, usages);
    }

    // import private key
    auto privateExponent = base64URLDecode(keyData.d);
    if (!privateExponent)
        return nullptr;
    if (keyData.p.isNull() && keyData.q.isNull() && keyData.dp.isNull() && keyData.dp.isNull() && keyData.qi.isNull()) {
        auto privateKeyComponents = CryptoKeyRSAComponents::createPrivate(WTFMove(*modulus), WTFMove(*exponent), WTFMove(*privateExponent));
        // Notice: CryptoAlgorithmIdentifier::SHA_1 is just a placeholder. It should not have any effect if hash is std::nullopt.
        return CryptoKeyRSA::create(algorithm, hash.value_or(CryptoAlgorithmIdentifier::SHA_1), !!hash, *privateKeyComponents, extractable, usages);
    }

    if (keyData.p.isNull() || keyData.q.isNull() || keyData.dp.isNull() || keyData.dq.isNull() || keyData.qi.isNull())
        return nullptr;
    
    auto firstPrimeFactor = base64URLDecode(keyData.p);
    if (!firstPrimeFactor)
        return nullptr;
    auto firstFactorCRTExponent = base64URLDecode(keyData.dp);
    if (!firstFactorCRTExponent)
        return nullptr;
    auto secondPrimeFactor = base64URLDecode(keyData.q);
    if (!secondPrimeFactor)
        return nullptr;
    auto secondFactorCRTExponent = base64URLDecode(keyData.dq);
    if (!secondFactorCRTExponent)
        return nullptr;
    auto secondFactorCRTCoefficient = base64URLDecode(keyData.qi);
    if (!secondFactorCRTCoefficient)
        return nullptr;

    CryptoKeyRSAComponents::PrimeInfo firstPrimeInfo;
    firstPrimeInfo.primeFactor = WTFMove(*firstPrimeFactor);
    firstPrimeInfo.factorCRTExponent = WTFMove(*firstFactorCRTExponent);
    
    CryptoKeyRSAComponents::PrimeInfo secondPrimeInfo;
    secondPrimeInfo.primeFactor = WTFMove(*secondPrimeFactor);
    secondPrimeInfo.factorCRTExponent = WTFMove(*secondFactorCRTExponent);
    secondPrimeInfo.factorCRTCoefficient = WTFMove(*secondFactorCRTCoefficient);

    if (!keyData.oth) {
        auto privateKeyComponents = CryptoKeyRSAComponents::createPrivateWithAdditionalData(WTFMove(*modulus), WTFMove(*exponent), WTFMove(*privateExponent), WTFMove(firstPrimeInfo), WTFMove(secondPrimeInfo), { });
        // Notice: CryptoAlgorithmIdentifier::SHA_1 is just a placeholder. It should not have any effect if hash is std::nullopt.
        return CryptoKeyRSA::create(algorithm, hash.value_or(CryptoAlgorithmIdentifier::SHA_1), !!hash, *privateKeyComponents, extractable, usages);
    }

    Vector<CryptoKeyRSAComponents::PrimeInfo> otherPrimeInfos;
    for (const auto& value : keyData.oth.value()) {
        auto primeFactor = base64URLDecode(value.r);
        if (!primeFactor)
            return nullptr;
        auto factorCRTExponent = base64URLDecode(value.d);
        if (!factorCRTExponent)
            return nullptr;
        auto factorCRTCoefficient = base64URLDecode(value.t);
        if (!factorCRTCoefficient)
            return nullptr;

        CryptoKeyRSAComponents::PrimeInfo info;
        info.primeFactor = WTFMove(*primeFactor);
        info.factorCRTExponent = WTFMove(*factorCRTExponent);
        info.factorCRTCoefficient = WTFMove(*factorCRTCoefficient);

        otherPrimeInfos.append(WTFMove(info));
    }

    auto privateKeyComponents = CryptoKeyRSAComponents::createPrivateWithAdditionalData(WTFMove(*modulus), WTFMove(*exponent), WTFMove(*privateExponent), WTFMove(firstPrimeInfo), WTFMove(secondPrimeInfo), WTFMove(otherPrimeInfos));
    // Notice: CryptoAlgorithmIdentifier::SHA_1 is just a placeholder. It should not have any effect if hash is std::nullopt.
    return CryptoKeyRSA::create(algorithm, hash.value_or(CryptoAlgorithmIdentifier::SHA_1), !!hash, *privateKeyComponents, extractable, usages);
}

JsonWebKey CryptoKeyRSA::exportJwk() const
{
    JsonWebKey result;
    result.kty = "RSA"_s;
    result.key_ops = usages();
    result.usages = usagesBitmap();
    result.ext = extractable();

    auto rsaComponents = exportData();

    if (!rsaComponents)
        return result;

    // public key
    result.n = base64URLEncodeToString(rsaComponents->modulus());
    result.e = base64URLEncodeToString(rsaComponents->exponent());
    if (rsaComponents->type() == CryptoKeyRSAComponents::Type::Public)
        return result;

    // private key
    result.d = base64URLEncodeToString(rsaComponents->privateExponent());
    if (!rsaComponents->hasAdditionalPrivateKeyParameters())
        return result;

    result.p = base64URLEncodeToString(rsaComponents->firstPrimeInfo().primeFactor);
    result.q = base64URLEncodeToString(rsaComponents->secondPrimeInfo().primeFactor);
    result.dp = base64URLEncodeToString(rsaComponents->firstPrimeInfo().factorCRTExponent);
    result.dq = base64URLEncodeToString(rsaComponents->secondPrimeInfo().factorCRTExponent);
    result.qi = base64URLEncodeToString(rsaComponents->secondPrimeInfo().factorCRTCoefficient);
    if (rsaComponents->otherPrimeInfos().isEmpty())
        return result;

    Vector<RsaOtherPrimesInfo> oth;
    for (const auto& info : rsaComponents->otherPrimeInfos()) {
        RsaOtherPrimesInfo otherInfo;
        otherInfo.r = base64URLEncodeToString(info.primeFactor);
        otherInfo.d = base64URLEncodeToString(info.factorCRTExponent);
        otherInfo.t = base64URLEncodeToString(info.factorCRTCoefficient);
        oth.append(WTFMove(otherInfo));
    }
    result.oth = WTFMove(oth);
    return result;
}

CryptoKey::Data CryptoKeyRSA::data() const
{
    auto jwk = exportJwk();
    std::optional<CryptoAlgorithmIdentifier> hash;
    if (m_restrictedToSpecificHash)
        hash = hashAlgorithmIdentifier();
    return CryptoKey::Data {
        CryptoKeyClass::RSA,
        algorithmIdentifier(),
        extractable(),
        usagesBitmap(),
        std::nullopt,
        WTFMove(jwk),
        hash
    };
}

} // namespace WebCore
