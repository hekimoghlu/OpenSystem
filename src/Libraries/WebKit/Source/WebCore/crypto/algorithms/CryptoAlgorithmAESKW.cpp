/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 26, 2025.
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
#include "CryptoAlgorithmAESKW.h"

#include "CryptoAlgorithmAesKeyParams.h"
#include "CryptoKeyAES.h"
#include <variant>

namespace WebCore {

namespace CryptoAlgorithmAESKWInternal {
static constexpr auto ALG128 = "A128KW"_s;
static constexpr auto ALG192 = "A192KW"_s;
static constexpr auto ALG256 = "A256KW"_s;
}

static inline bool usagesAreInvalidForCryptoAlgorithmAESKW(CryptoKeyUsageBitmap usages)
{
    return usages & (CryptoKeyUsageSign | CryptoKeyUsageVerify | CryptoKeyUsageDeriveKey | CryptoKeyUsageDeriveBits | CryptoKeyUsageEncrypt | CryptoKeyUsageDecrypt);
}

Ref<CryptoAlgorithm> CryptoAlgorithmAESKW::create()
{
    return adoptRef(*new CryptoAlgorithmAESKW);
}

CryptoAlgorithmIdentifier CryptoAlgorithmAESKW::identifier() const
{
    return s_identifier;
}

void CryptoAlgorithmAESKW::generateKey(const CryptoAlgorithmParameters& parameters, bool extractable, CryptoKeyUsageBitmap usages, KeyOrKeyPairCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&)
{
    if (usagesAreInvalidForCryptoAlgorithmAESKW(usages)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }

    auto result = CryptoKeyAES::generate(CryptoAlgorithmIdentifier::AES_KW, downcast<CryptoAlgorithmAesKeyParams>(parameters).length, extractable, usages);
    if (!result) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    callback(WTFMove(result));
}

void CryptoAlgorithmAESKW::importKey(CryptoKeyFormat format, KeyData&& data, const CryptoAlgorithmParameters& parameters, bool extractable, CryptoKeyUsageBitmap usages, KeyCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    using namespace CryptoAlgorithmAESKWInternal;

    if (usagesAreInvalidForCryptoAlgorithmAESKW(usages)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }

    RefPtr<CryptoKeyAES> result;
    switch (format) {
    case CryptoKeyFormat::Raw:
        result = CryptoKeyAES::importRaw(parameters.identifier, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    case CryptoKeyFormat::Jwk: {
        result = CryptoKeyAES::importJwk(parameters.identifier, WTFMove(std::get<JsonWebKey>(data)), extractable, usages, [](size_t length, const String& alg) -> bool {
            switch (length) {
            case CryptoKeyAES::s_length128:
                return alg.isNull() || alg == ALG128;
            case CryptoKeyAES::s_length192:
                return alg.isNull() || alg == ALG192;
            case CryptoKeyAES::s_length256:
                return alg.isNull() || alg == ALG256;
            }
            return false;
        });
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

void CryptoAlgorithmAESKW::exportKey(CryptoKeyFormat format, Ref<CryptoKey>&& key, KeyDataCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    using namespace CryptoAlgorithmAESKWInternal;
    const auto& aesKey = downcast<CryptoKeyAES>(key.get());

    if (aesKey.key().isEmpty()) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    KeyData result;
    switch (format) {
    case CryptoKeyFormat::Raw:
        result = Vector<uint8_t>(aesKey.key());
        break;
    case CryptoKeyFormat::Jwk: {
        JsonWebKey jwk = aesKey.exportJwk();
        switch (aesKey.key().size() * 8) {
        case CryptoKeyAES::s_length128:
            jwk.alg = String(ALG128);
            break;
        case CryptoKeyAES::s_length192:
            jwk.alg = String(ALG192);
            break;
        case CryptoKeyAES::s_length256:
            jwk.alg = String(ALG256);
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

void CryptoAlgorithmAESKW::wrapKey(Ref<CryptoKey>&& key, Vector<uint8_t>&& data, VectorCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    if (data.size() % 8) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }
    auto result = platformWrapKey(downcast<CryptoKeyAES>(key.get()), WTFMove(data));
    if (result.hasException()) {
        exceptionCallback(result.releaseException().code());
        return;
    }

    callback(result.releaseReturnValue());
}

void CryptoAlgorithmAESKW::unwrapKey(Ref<CryptoKey>&& key, Vector<uint8_t>&& data, VectorCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    auto result = platformUnwrapKey(downcast<CryptoKeyAES>(key.get()), WTFMove(data));
    if (result.hasException()) {
        exceptionCallback(result.releaseException().code());
        return;
    }

    callback(result.releaseReturnValue());
}

ExceptionOr<std::optional<size_t>> CryptoAlgorithmAESKW::getKeyLength(const CryptoAlgorithmParameters& parameters)
{
    return CryptoKeyAES::getKeyLength(parameters);
}

} // namespace WebCore
