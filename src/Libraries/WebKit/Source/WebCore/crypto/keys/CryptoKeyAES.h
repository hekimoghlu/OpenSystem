/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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
#include "CryptoKey.h"
#include "ExceptionOr.h"
#include <wtf/Function.h>
#include <wtf/Vector.h>

namespace WebCore {

class CryptoAlgorithmParameters;

struct JsonWebKey;

class CryptoKeyAES final : public CryptoKey {
public:
    static const int s_length128 = 128;
    static const int s_length192 = 192;
    static const int s_length256 = 256;

    static Ref<CryptoKeyAES> create(CryptoAlgorithmIdentifier algorithm, const Vector<uint8_t>& key, bool extractable, CryptoKeyUsageBitmap usage)
    {
        return adoptRef(*new CryptoKeyAES(algorithm, key, extractable, usage));
    }
    virtual ~CryptoKeyAES();

    static bool isValidAESAlgorithm(CryptoAlgorithmIdentifier);

    static RefPtr<CryptoKeyAES> generate(CryptoAlgorithmIdentifier, size_t lengthBits, bool extractable, CryptoKeyUsageBitmap);
    WEBCORE_EXPORT static RefPtr<CryptoKeyAES> importRaw(CryptoAlgorithmIdentifier, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap);
    using CheckAlgCallback = Function<bool(size_t, const String&)>;
    static RefPtr<CryptoKeyAES> importJwk(CryptoAlgorithmIdentifier, JsonWebKey&&, bool extractable, CryptoKeyUsageBitmap, CheckAlgCallback&&);

    CryptoKeyClass keyClass() const final { return CryptoKeyClass::AES; }

    const Vector<uint8_t>& key() const { return m_key; }
    JsonWebKey exportJwk() const;

    static ExceptionOr<std::optional<size_t>> getKeyLength(const CryptoAlgorithmParameters&);

private:
    CryptoKeyAES(CryptoAlgorithmIdentifier, const Vector<uint8_t>& key, bool extractable, CryptoKeyUsageBitmap);
    CryptoKeyAES(CryptoAlgorithmIdentifier, Vector<uint8_t>&& key, bool extractable, CryptoKeyUsageBitmap);

    KeyAlgorithm algorithm() const final;
    CryptoKey::Data data() const final;

    Vector<uint8_t> m_key;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CRYPTO_KEY(CryptoKeyAES, CryptoKeyClass::AES)
