/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#include "CryptoKeyHMAC.h"

#include "CryptoAlgorithmHmacKeyParams.h"
#include "CryptoAlgorithmRegistry.h"
#include "ExceptionOr.h"
#include "JsonWebKey.h"
#include <wtf/text/Base64.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

static std::optional<size_t> getKeyLengthFromHash(CryptoAlgorithmIdentifier hash)
{
    switch (hash) {
    case CryptoAlgorithmIdentifier::SHA_1:
    case CryptoAlgorithmIdentifier::SHA_224:
    case CryptoAlgorithmIdentifier::SHA_256:
        return 512;
    case CryptoAlgorithmIdentifier::SHA_384:
    case CryptoAlgorithmIdentifier::SHA_512:
        return 1024;
    default:
        RELEASE_ASSERT_NOT_REACHED_WITH_MESSAGE("Invalid Hash Algorithm");
        return { };
    }
}

CryptoKeyHMAC::CryptoKeyHMAC(const Vector<uint8_t>& key, CryptoAlgorithmIdentifier hash, bool extractable, CryptoKeyUsageBitmap usage)
    : CryptoKey(CryptoAlgorithmIdentifier::HMAC, CryptoKeyType::Secret, extractable, usage)
    , m_hash(hash)
    , m_key(key)
{
}

CryptoKeyHMAC::CryptoKeyHMAC(Vector<uint8_t>&& key, CryptoAlgorithmIdentifier hash, bool extractable, CryptoKeyUsageBitmap usage)
    : CryptoKey(CryptoAlgorithmIdentifier::HMAC, CryptoKeyType::Secret, extractable, usage)
    , m_hash(hash)
    , m_key(WTFMove(key))
{
}

CryptoKeyHMAC::~CryptoKeyHMAC() = default;

RefPtr<CryptoKeyHMAC> CryptoKeyHMAC::generate(size_t lengthBits, CryptoAlgorithmIdentifier hash, bool extractable, CryptoKeyUsageBitmap usages)
{
    if (!lengthBits) {
        auto length = getKeyLengthFromHash(hash);
        if (!length.value_or(0))
            return nullptr;
        lengthBits = *length;
    }
    // CommonHMAC only supports key length that is a multiple of 8. Therefore, here we are a little bit different
    // from the spec as of 11 December 2014: https://www.w3.org/TR/WebCryptoAPI/#hmac-operations
    if (lengthBits % 8)
        return nullptr;

    return adoptRef(new CryptoKeyHMAC(randomData(lengthBits / 8), hash, extractable, usages));
}

RefPtr<CryptoKeyHMAC> CryptoKeyHMAC::importRaw(size_t lengthBits, CryptoAlgorithmIdentifier hash, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap usages)
{
    size_t length = keyData.size() * 8;
    if (!length)
        return nullptr;
    // CommonHMAC only supports key length that is a multiple of 8. Therefore, here we are a little bit different
    // from the spec as of 11 December 2014: https://www.w3.org/TR/WebCryptoAPI/#hmac-operations
    if (lengthBits && lengthBits != length)
        return nullptr;

    return adoptRef(new CryptoKeyHMAC(WTFMove(keyData), hash, extractable, usages));
}

RefPtr<CryptoKeyHMAC> CryptoKeyHMAC::importJwk(size_t lengthBits, CryptoAlgorithmIdentifier hash, JsonWebKey&& keyData, bool extractable, CryptoKeyUsageBitmap usages, CheckAlgCallback&& callback)
{
    if (keyData.kty != "oct"_s)
        return nullptr;
    if (keyData.k.isNull())
        return nullptr;
    auto octetSequence = base64URLDecode(keyData.k);
    if (!octetSequence)
        return nullptr;
    if (!callback(hash, keyData.alg))
        return nullptr;
    if (usages && !keyData.use.isNull() && keyData.use != "sig"_s)
        return nullptr;
    if (keyData.key_ops && ((keyData.usages & usages) != usages))
        return nullptr;
    if (keyData.ext && !keyData.ext.value() && extractable)
        return nullptr;

    return CryptoKeyHMAC::importRaw(lengthBits, hash, WTFMove(*octetSequence), extractable, usages);
}

JsonWebKey CryptoKeyHMAC::exportJwk() const
{
    JsonWebKey result;
    result.kty = "oct"_s;
    result.k = base64URLEncodeToString(m_key);
    result.key_ops = usages();
    result.usages = usagesBitmap();
    result.ext = extractable();
    return result;
}

ExceptionOr<std::optional<size_t>> CryptoKeyHMAC::getKeyLength(const CryptoAlgorithmParameters& parameters)
{
    auto& aesParameters = downcast<CryptoAlgorithmHmacKeyParams>(parameters);

    auto length = aesParameters.length ? *(aesParameters.length) : getKeyLengthFromHash(aesParameters.hashIdentifier);
    if (!length.value_or(0))
        return Exception { ExceptionCode::TypeError };
    return length;
}

auto CryptoKeyHMAC::algorithm() const -> KeyAlgorithm
{
    CryptoHmacKeyAlgorithm result;
    result.name = CryptoAlgorithmRegistry::singleton().name(algorithmIdentifier());
    result.hash.name = CryptoAlgorithmRegistry::singleton().name(m_hash);
    result.length = m_key.size() * 8;
    return result;
}

CryptoKey::Data CryptoKeyHMAC::data() const
{
    auto keyData = key();
    return CryptoKey::Data {
        CryptoKeyClass::HMAC,
        algorithmIdentifier(),
        extractable(),
        usagesBitmap(),
        std::nullopt,
        exportJwk(),
        hashAlgorithmIdentifier(),
        std::nullopt,
        key().size() * CHAR_BIT, // Size in bits.
    };
}

} // namespace WebCore
