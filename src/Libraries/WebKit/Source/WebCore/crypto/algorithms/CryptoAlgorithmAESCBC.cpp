/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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
#include "CryptoAlgorithmAESCBC.h"

#include "CryptoAlgorithmAesCbcCfbParams.h"
#include "CryptoAlgorithmAesKeyParams.h"
#include "CryptoKeyAES.h"
#include <wtf/CrossThreadCopier.h>

namespace WebCore {

namespace CryptoAlgorithmAESCBCInternal {
static constexpr auto ALG128 = "A128CBC"_s;
static constexpr auto ALG192 = "A192CBC"_s;
static constexpr auto ALG256 = "A256CBC"_s;
static const size_t IVSIZE = 16;
}

static inline bool usagesAreInvalidForCryptoAlgorithmAESCBC(CryptoKeyUsageBitmap usages)
{
    return usages & (CryptoKeyUsageSign | CryptoKeyUsageVerify | CryptoKeyUsageDeriveKey | CryptoKeyUsageDeriveBits);
}

Ref<CryptoAlgorithm> CryptoAlgorithmAESCBC::create()
{
    return adoptRef(*new CryptoAlgorithmAESCBC);
}

CryptoAlgorithmIdentifier CryptoAlgorithmAESCBC::identifier() const
{
    return s_identifier;
}

void CryptoAlgorithmAESCBC::encrypt(const CryptoAlgorithmParameters& parameters, Ref<CryptoKey>&& key, Vector<uint8_t>&& plainText, VectorCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    using namespace CryptoAlgorithmAESCBCInternal;

    auto& aesParameters = downcast<CryptoAlgorithmAesCbcCfbParams>(parameters);
    if (aesParameters.ivVector().size() != IVSIZE) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    dispatchOperationInWorkQueue(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback),
        [parameters = crossThreadCopy(aesParameters), key = WTFMove(key), plainText = WTFMove(plainText)] {
            return platformEncrypt(parameters, downcast<CryptoKeyAES>(key.get()), plainText);
        });
}

void CryptoAlgorithmAESCBC::decrypt(const CryptoAlgorithmParameters& parameters, Ref<CryptoKey>&& key, Vector<uint8_t>&& cipherText, VectorCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext& context, WorkQueue& workQueue)
{
    using namespace CryptoAlgorithmAESCBCInternal;

    auto& aesParameters = downcast<CryptoAlgorithmAesCbcCfbParams>(parameters);
    if (aesParameters.ivVector().size() != IVSIZE) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    dispatchOperationInWorkQueue(workQueue, context, WTFMove(callback), WTFMove(exceptionCallback),
        [parameters = crossThreadCopy(aesParameters), key = WTFMove(key), cipherText = WTFMove(cipherText)] {
            return platformDecrypt(parameters, downcast<CryptoKeyAES>(key.get()), cipherText);
        });
}

void CryptoAlgorithmAESCBC::generateKey(const CryptoAlgorithmParameters& parameters, bool extractable, CryptoKeyUsageBitmap usages, KeyOrKeyPairCallback&& callback, ExceptionCallback&& exceptionCallback, ScriptExecutionContext&)
{
    const auto& aesParameters = downcast<CryptoAlgorithmAesKeyParams>(parameters);

    if (usagesAreInvalidForCryptoAlgorithmAESCBC(usages)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }

    auto result = CryptoKeyAES::generate(CryptoAlgorithmIdentifier::AES_CBC, aesParameters.length, extractable, usages);
    if (!result) {
        exceptionCallback(ExceptionCode::OperationError);
        return;
    }

    callback(WTFMove(result));
}

void CryptoAlgorithmAESCBC::importKey(CryptoKeyFormat format, KeyData&& data, const CryptoAlgorithmParameters& parameters, bool extractable, CryptoKeyUsageBitmap usages, KeyCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    using namespace CryptoAlgorithmAESCBCInternal;

    if (usagesAreInvalidForCryptoAlgorithmAESCBC(usages)) {
        exceptionCallback(ExceptionCode::SyntaxError);
        return;
    }

    RefPtr<CryptoKeyAES> result;
    switch (format) {
    case CryptoKeyFormat::Raw:
        result = CryptoKeyAES::importRaw(parameters.identifier, WTFMove(std::get<Vector<uint8_t>>(data)), extractable, usages);
        break;
    case CryptoKeyFormat::Jwk: {
        auto checkAlgCallback = [](size_t length, const String& alg) -> bool {
            switch (length) {
            case CryptoKeyAES::s_length128:
                return alg.isNull() || alg == ALG128;
            case CryptoKeyAES::s_length192:
                return alg.isNull() || alg == ALG192;
            case CryptoKeyAES::s_length256:
                return alg.isNull() || alg == ALG256;
            }
            return false;
        };
        result = CryptoKeyAES::importJwk(parameters.identifier, WTFMove(std::get<JsonWebKey>(data)), extractable, usages, WTFMove(checkAlgCallback));
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

void CryptoAlgorithmAESCBC::exportKey(CryptoKeyFormat format, Ref<CryptoKey>&& key, KeyDataCallback&& callback, ExceptionCallback&& exceptionCallback)
{
    using namespace CryptoAlgorithmAESCBCInternal;
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

ExceptionOr<std::optional<size_t>> CryptoAlgorithmAESCBC::getKeyLength(const CryptoAlgorithmParameters& parameters)
{
    return CryptoKeyAES::getKeyLength(parameters);
}

} // namespace WebCore
