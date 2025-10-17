/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include "CryptoKeyAES.h"

#include "CryptoAesKeyAlgorithm.h"
#include "CryptoAlgorithmAesKeyParams.h"
#include "CryptoAlgorithmRegistry.h"
#include "ExceptionOr.h"
#include "JsonWebKey.h"
#include <wtf/text/Base64.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

static inline bool lengthIsValid(size_t length)
{
    return (length == CryptoKeyAES::s_length128) || (length == CryptoKeyAES::s_length192) || (length == CryptoKeyAES::s_length256);
}

CryptoKeyAES::CryptoKeyAES(CryptoAlgorithmIdentifier algorithm, const Vector<uint8_t>& key, bool extractable, CryptoKeyUsageBitmap usage)
    : CryptoKey(algorithm, CryptoKeyType::Secret, extractable, usage)
    , m_key(key)
{
    ASSERT(isValidAESAlgorithm(algorithm));
}

CryptoKeyAES::CryptoKeyAES(CryptoAlgorithmIdentifier algorithm, Vector<uint8_t>&& key, bool extractable, CryptoKeyUsageBitmap usage)
    : CryptoKey(algorithm, CryptoKeyType::Secret, extractable, usage)
    , m_key(WTFMove(key))
{
    ASSERT(isValidAESAlgorithm(algorithm));
}

CryptoKeyAES::~CryptoKeyAES() = default;

bool CryptoKeyAES::isValidAESAlgorithm(CryptoAlgorithmIdentifier algorithm)
{
    return algorithm == CryptoAlgorithmIdentifier::AES_CTR
        || algorithm == CryptoAlgorithmIdentifier::AES_CBC
        || algorithm == CryptoAlgorithmIdentifier::AES_GCM
        || algorithm == CryptoAlgorithmIdentifier::AES_CFB
        || algorithm == CryptoAlgorithmIdentifier::AES_KW;
}

RefPtr<CryptoKeyAES> CryptoKeyAES::generate(CryptoAlgorithmIdentifier algorithm, size_t lengthBits, bool extractable, CryptoKeyUsageBitmap usages)
{
    if (!lengthIsValid(lengthBits))
        return nullptr;
    return adoptRef(new CryptoKeyAES(algorithm, randomData(lengthBits / 8), extractable, usages));
}

RefPtr<CryptoKeyAES> CryptoKeyAES::importRaw(CryptoAlgorithmIdentifier algorithm, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    if (!lengthIsValid(keyData.size() * 8))
        return nullptr;
    return adoptRef(new CryptoKeyAES(algorithm, WTFMove(keyData), extractable, usages));
}

RefPtr<CryptoKeyAES> CryptoKeyAES::importJwk(CryptoAlgorithmIdentifier algorithm, JsonWebKey&& keyData, bool extractable, CryptoKeyUsageBitmap usages, CheckAlgCallback&& callback)
{
    if (keyData.kty != "oct"_s)
        return nullptr;
    if (keyData.k.isNull())
        return nullptr;
    auto octetSequence = base64URLDecode(keyData.k);
    if (!octetSequence)
        return nullptr;
    if (!callback(octetSequence->size() * 8, keyData.alg))
        return nullptr;
    if (usages && !keyData.use.isNull() && keyData.use != "enc"_s)
        return nullptr;
    if (keyData.key_ops && ((keyData.usages & usages) != usages))
        return nullptr;
    if (keyData.ext && !keyData.ext.value() && extractable)
        return nullptr;

    return adoptRef(new CryptoKeyAES(algorithm, WTFMove(*octetSequence), extractable, usages));
}

JsonWebKey CryptoKeyAES::exportJwk() const
{
    JsonWebKey result;
    result.kty = "oct"_s;
    result.k = base64URLEncodeToString(m_key);
    result.key_ops = usages();
    result.usages = usagesBitmap();
    result.ext = extractable();
    return result;
}

ExceptionOr<std::optional<size_t>> CryptoKeyAES::getKeyLength(const CryptoAlgorithmParameters& parameters)
{
    auto& aesParameters = downcast<CryptoAlgorithmAesKeyParams>(parameters);
    if (!lengthIsValid(aesParameters.length))
        return Exception { ExceptionCode::OperationError };
    return aesParameters.length;
}

auto CryptoKeyAES::algorithm() const -> KeyAlgorithm
{
    CryptoAesKeyAlgorithm result;
    result.name = CryptoAlgorithmRegistry::singleton().name(algorithmIdentifier());
    result.length = m_key.size() * 8;
    return result;
}

CryptoKey::Data CryptoKeyAES::data() const
{
    return CryptoKey::Data {
        CryptoKeyClass::AES,
        algorithmIdentifier(),
        extractable(),
        usagesBitmap(),
        std::nullopt,
        exportJwk()
    };
}

} // namespace WebCore
