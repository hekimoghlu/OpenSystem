/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#include "CryptoAlgorithmHMAC.h"

#include "CryptoAlgorithmHmacKeyParams.h"
#include "CryptoKeyHMAC.h"
#include "ScriptExecutionContext.h"
#include <variant>

namespace WebCore {

namespace CryptoAlgorithmHMACInternal {
static constexpr auto ALG1 = "HS1"_s;
static constexpr auto ALG224 = "HS224"_s;
static constexpr auto ALG256 = "HS256"_s;
static constexpr auto ALG384 = "HS384"_s;
static constexpr auto ALG512 = "HS512"_s;
}

static inline bool usagesAreInvalidForCryptoAlgorithmHMAC(CryptoKeyUsageBitmap usages)
{
    return usages & (CryptoKeyUsageEncrypt | CryptoKeyUsageDecrypt | CryptoKeyUsageDeriveKey | CryptoKeyUsageDeriveBits | CryptoKeyUsageWrapKey | CryptoKeyUsageUnwrapKey);
}

Ref<CryptoAlgorithm> CryptoAlgorithmHMAC::create()
{
    return adoptRef(*new CryptoAlgorithmHMAC);
}

CryptoAlgorithmIdentifier CryptoAlgorithmHMAC::identifier() const
{
    return s_identifier;
}

void CryptoAlgorithmHMAC::sign(const CryptoAlgorithmParameters&, Ref<CryptoKey>&& key, Vector<uint8_t>&& data, VectorCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    dispatchOperationInWorkQueue(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback),
        [key = WTFMove(key), data = WTFMove(data)] {
            return platformSign(downcast<CryptoKeyHMAC>(key.get()), data);
        });
}

void CryptoAlgorithmHMAC::verify(const CryptoAlgorithmParameters&, Ref<CryptoKey>&& key, Vector<uint8_t>&& signature, Vector<uint8_t>&& data, BoolCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    dispatchOperationInWorkQueue(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback),
        [key = WTFMove(key), signature = WTFMove(signature), data = WTFMove(data)] {
            return platformVerify(downcast<CryptoKeyHMAC>(key.get()), signature, data);
        });
}

void CryptoAlgorithmHMAC::generateKey(const CryptoAlgorithmParameters& parameters, bool extractable, CryptoKeyUsageBitmap usages, KeyOrKeyPairCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&)
{
    const auto& hmacParameters = downcast<CryptoAlgorithmHmacKeyParams>(parameters);

    if (usagesAreInvalidForCryptoAlgorithmHMAC(usages)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }

    if (hmacParameters.length && !hmacParameters.length.value()) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    auto result = CryptoKeyHMAC::generate(hmacParameters.length.value_or(0), hmacParameters.hashIdentifier, extractable, usages);
    if (!result) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    callback(WTFMove(result));
}

void CryptoAlgorithmHMAC::importKey(CryptoKeyFormat format, KeyData&& data, const CryptoAlgorithmParameters& parameters, bool extractable, CryptoKeyUsageBitmap usages, KeyCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    using namespace CryptoAlgorithmHMACInternal;

    const auto& hmacParameters = downcast<CryptoAlgorithmHmacKeyParams>(parameters);

    if (usagesAreInvalidForCryptoAlgorithmHMAC(usages)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }

    RefPtr<CryptoKeyHMAC> result;
    switch (format) {
    case CryptoKeyFormat::Raw:
        result = CryptoKeyHMAC::importRaw(hmacParameters.length.value_or(0), hmacParameters.hashIdentifier, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    case CryptoKeyFormat::Jwk: {
        auto checkAlgCallback = [](CryptoAlgorithmIdentifier hash, const String& alg) -> bool {
            switch (hash) {
            case CryptoAlgorithmIdentifier::SHA_1:
                return alg.isNull() || alg == ALG1;
            case CryptoAlgorithmIdentifier::SHA_224:
                return alg.isNull() || alg == ALG224;
            case CryptoAlgorithmIdentifier::SHA_256:
                return alg.isNull() || alg == ALG256;
            case CryptoAlgorithmIdentifier::SHA_384:
                return alg.isNull() || alg == ALG384;
            case CryptoAlgorithmIdentifier::SHA_512:
                return alg.isNull() || alg == ALG512;
            default:
                return false;
            }
            return false;
        };
        result = CryptoKeyHMAC::importJwk(hmacParameters.length.value_or(0), hmacParameters.hashIdentifier, WTFMove(std::get<JsonWebKey>(data)), extractable, usages, WTFMove(checkAlgCallback));
        break;
    }
    default:
        exceptionCallback(ExceptionCode::NotSupportedError);
        return;
    }
    if (!result) {
        exceptionCallback(ExceptionCode::DataError);
        return;
    }

    callback(*result);
}

void CryptoAlgorithmHMAC::exportKey(CryptoKeyFormat format, Ref<CryptoKey>&& key, KeyDataCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    using namespace CryptoAlgorithmHMACInternal;
    const auto& hmacKey = downcast<CryptoKeyHMAC>(key.get());

    if (hmacKey.key().isEmpty()) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    KeyData result;
    switch (format) {
    case CryptoKeyFormat::Raw:
        result = Vector<uint8_t>(hmacKey.key());
        break;
    case CryptoKeyFormat::Jwk: {
        JsonWebKey jwk = hmacKey.exportJwk();
        switch (hmacKey.hashAlgorithmIdentifier()) {
        case CryptoAlgorithmIdentifier::SHA_1:
            jwk.alg = String(ALG1);
            break;
        case CryptoAlgorithmIdentifier::SHA_224:
            jwk.alg = String(ALG224);
            break;
        case CryptoAlgorithmIdentifier::SHA_256:
            jwk.alg = String(ALG256);
            break;
        case CryptoAlgorithmIdentifier::SHA_384:
            jwk.alg = String(ALG384);
            break;
        case CryptoAlgorithmIdentifier::SHA_512:
            jwk.alg = String(ALG512);
            break;
        default:
            ASSERT_NOT_REACHED();
        }
        result = WTFMove(jwk);
        break;
    }
    default:
        exceptionCallback(ExceptionCode::NotSupportedError);
        return;
    }

    callback(format, WTFMove(result));
}

ExceptionOr<std::optional<size_t>> CryptoAlgorithmHMAC::getKeyLength(const CryptoAlgorithmParameters& parameters)
{
    return CryptoKeyHMAC::getKeyLength(parameters);
}

} // namespace WebCore
