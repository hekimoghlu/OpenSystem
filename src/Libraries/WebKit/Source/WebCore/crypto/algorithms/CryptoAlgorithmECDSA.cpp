/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
#include "CryptoAlgorithmECDSA.h"

#include "CryptoAlgorithmEcKeyParams.h"
#include "CryptoAlgorithmEcdsaParams.h"
#include "CryptoKeyEC.h"
#include "ScriptExecutionContext.h"
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <wtf/CrossThreadCopier.h>

namespace WebCore {

namespace CryptoAlgorithmECDSAInternal {
static constexpr auto ALG256 = "ES256"_s;
static constexpr auto ALG384 = "ES384"_s;
static constexpr auto ALG512 = "ES512"_s;
static constexpr auto P256 = "P-256"_s;
static constexpr auto P384 = "P-384"_s;
static constexpr auto P521 = "P-521"_s;
}

Ref<CryptoAlgorithm> CryptoAlgorithmECDSA::create()
{
    return adoptRef(*new CryptoAlgorithmECDSA);
}

CryptoAlgorithmIdentifier CryptoAlgorithmECDSA::identifier() const
{
    return s_identifier;
}

void CryptoAlgorithmECDSA::sign(const CryptoAlgorithmParameters& parameters, Ref<CryptoKey>&& key, Vector<uint8_t>&& data, VectorCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    if (key->type() != CryptoKeyType::Private) {
        exceptionCallback(ExceptionCode::InvalidAccessError);
        return;
    }
    dispatchOperationInWorkQueue(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback),
        [parameters = crossThreadCopy(downcast<CryptoAlgorithmEcdsaParams>(parameters)), key = WTFMove(key), data = WTFMove(data)] {
            return platformSign(parameters, downcast<CryptoKeyEC>(key.get()), data);
        });
}

void CryptoAlgorithmECDSA::verify(const CryptoAlgorithmParameters& parameters, Ref<CryptoKey>&& key, Vector<uint8_t>&& signature, Vector<uint8_t>&& data, BoolCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    if (key->type() != CryptoKeyType::Public) {
        exceptionCallback(ExceptionCode::InvalidAccessError);
        return;
    }

    dispatchOperationInWorkQueue(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback),
        [parameters = crossThreadCopy(downcast<CryptoAlgorithmEcdsaParams>(parameters)), key = WTFMove(key), signature = WTFMove(signature), data = WTFMove(data)] {
            return platformVerify(parameters, downcast<CryptoKeyEC>(key.get()), signature, data);
        });
}

void CryptoAlgorithmECDSA::generateKey(const CryptoAlgorithmParameters& parameters, bool extractable, CryptoKeyUsageBitmap usages, KeyOrKeyPairCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&)
{
    const auto& ecParameters = downcast<CryptoAlgorithmEcKeyParams>(parameters);

    if (usages & (CryptoKeyUsageEncrypt | CryptoKeyUsageDecrypt | CryptoKeyUsageDeriveKey | CryptoKeyUsageDeriveBits | CryptoKeyUsageWrapKey | CryptoKeyUsageUnwrapKey)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }
    auto result = CryptoKeyEC::generatePair(CryptoAlgorithmIdentifier::ECDSA, ecParameters.namedCurve, extractable, usages);
    if (result.hasException()) {
        exceptionCallback(result.releaseException().code());
        return;
    }

    auto pair = result.releaseReturnValue();
    pair.publicKey->setUsagesBitmap(pair.publicKey->usagesBitmap() & CryptoKeyUsageVerify);
    pair.privateKey->setUsagesBitmap(pair.privateKey->usagesBitmap() & CryptoKeyUsageSign);
    callback(WTFMove(pair));
}

void CryptoAlgorithmECDSA::importKey(CryptoKeyFormat format, KeyData&& data, const CryptoAlgorithmParameters& parameters, bool extractable, CryptoKeyUsageBitmap usages, KeyCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    using namespace CryptoAlgorithmECDSAInternal;
    const auto& ecParameters = downcast<CryptoAlgorithmEcKeyParams>(parameters);

    RefPtr<CryptoKeyEC> result;
    switch (format) {
    case CryptoKeyFormat::Jwk: {
        JsonWebKey key = WTFMove(std::get<JsonWebKey>(data));

        if (usages && ((!key.d.isNull() && (usages ^ CryptoKeyUsageSign)) || (key.d.isNull() && (usages ^ CryptoKeyUsageVerify)))) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }
        if (usages && !key.use.isNull() && key.use != "sig"_s) {
            exceptionCallback(ExceptionCode::DataError);
            return;
        }

        bool isMatched = false;
        if (key.crv == P256)
            isMatched = key.alg.isNull() || key.alg == ALG256;
        if (key.crv == P384)
            isMatched = key.alg.isNull() || key.alg == ALG384;
        if (key.crv == P521)
            isMatched = key.alg.isNull() || key.alg == ALG512;
        if (!isMatched) {
            exceptionCallback(ExceptionCode::DataError);
            return;
        }

        result = CryptoKeyEC::importJwk(ecParameters.identifier, ecParameters.namedCurve, WTFMove(key), extractable, usages);
        break;
    }
    case CryptoKeyFormat::Raw:
        if (usages && (usages ^ CryptoKeyUsageVerify)) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }
        result = CryptoKeyEC::importRaw(ecParameters.identifier, ecParameters.namedCurve, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    case CryptoKeyFormat::Spki:
        if (usages && (usages ^ CryptoKeyUsageVerify)) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }
        result = CryptoKeyEC::importSpki(ecParameters.identifier, ecParameters.namedCurve, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    case CryptoKeyFormat::Pkcs8:
        if (usages && (usages ^ CryptoKeyUsageSign)) {
            exceptionCallback(ExceptionCode::SyntaxError);
            return;
        }
        result = CryptoKeyEC::importPkcs8(ecParameters.identifier, ecParameters.namedCurve, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    }
    if (!result) {
        exceptionCallback(ExceptionCode::DataError);
        return;
    }

    callback(*result);
}

void CryptoAlgorithmECDSA::exportKey(CryptoKeyFormat format, Ref<CryptoKey>&& key, KeyDataCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    const auto& ecKey = downcast<CryptoKeyEC>(key.get());
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
