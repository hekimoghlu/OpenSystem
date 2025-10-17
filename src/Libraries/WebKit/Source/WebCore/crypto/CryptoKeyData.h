/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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

#include "CryptoAlgorithmIdentifier.h"
#include "CryptoKeyType.h"
#include "CryptoKeyUsage.h"
#include "JsonWebKey.h"

namespace WebCore {

enum class CryptoKeyClass : uint8_t {
    AES,
    EC,
    HMAC,
    OKP,
    RSA,
    Raw
};

struct CryptoKeyData {
    CryptoKeyData(CryptoKeyClass keyClass, CryptoAlgorithmIdentifier algorithmIdentifier, bool extractable, CryptoKeyUsageBitmap usages, std::optional<Vector<uint8_t>> key, std::optional<JsonWebKey> jwk = std::nullopt, std::optional<CryptoAlgorithmIdentifier> hashAlgorithmIdentifier = std::nullopt, std::optional<String>&& namedCurveString = std::nullopt, std::optional<size_t> lengthBits = std::nullopt, std::optional<CryptoKeyType> type = std::nullopt)
        : keyClass(keyClass)
        , algorithmIdentifier(algorithmIdentifier)
        , extractable(extractable)
        , usages(usages)
        , key(WTFMove(key))
        , jwk(WTFMove(jwk))
        , hashAlgorithmIdentifier(hashAlgorithmIdentifier)
        , namedCurveString(WTFMove(namedCurveString))
        , lengthBits(lengthBits)
        , type(type)
    {
    }
    CryptoKeyData isolatedCopy() && {
        return {
            keyClass,
            algorithmIdentifier,
            extractable,
            usages,
            key,
            crossThreadCopy(WTFMove(jwk)),
            hashAlgorithmIdentifier,
            crossThreadCopy(WTFMove(namedCurveString)),
            lengthBits,
            type
        };
    }

    CryptoKeyClass keyClass;
    CryptoAlgorithmIdentifier algorithmIdentifier;
    bool extractable { false };
    CryptoKeyUsageBitmap usages { 0 };
    std::optional<Vector<uint8_t>> key;
    std::optional<JsonWebKey> jwk;
    std::optional<CryptoAlgorithmIdentifier> hashAlgorithmIdentifier;
    std::optional<String> namedCurveString;
    std::optional<size_t> lengthBits;
    std::optional<CryptoKeyType> type;
};

} // namespace WebCore
