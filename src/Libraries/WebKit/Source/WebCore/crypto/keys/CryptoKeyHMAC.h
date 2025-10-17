/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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

#include "CryptoKey.h"
#include "ExceptionOr.h"
#include <wtf/Function.h>
#include <wtf/Vector.h>

namespace WebCore {

class CryptoAlgorithmParameters;
struct JsonWebKey;

class CryptoKeyHMAC final : public CryptoKey {
public:
    static Ref<CryptoKeyHMAC> create(const Vector<uint8_t>& key, CryptoAlgorithmIdentifier hash, bool extractable, CryptoKeyUsageBitmap usage)
    {
        return adoptRef(*new CryptoKeyHMAC(key, hash, extractable, usage));
    }
    virtual ~CryptoKeyHMAC();

    static RefPtr<CryptoKeyHMAC> generate(size_t lengthBits, CryptoAlgorithmIdentifier hash, bool extractable, CryptoKeyUsageBitmap);
    static RefPtr<CryptoKeyHMAC> importRaw(size_t lengthBits, CryptoAlgorithmIdentifier hash, Vector<uint8_t>&& keyData, bool extractable, CryptoKeyUsageBitmap);
    using CheckAlgCallback = Function<bool(CryptoAlgorithmIdentifier, const String&)>;
    static RefPtr<CryptoKeyHMAC> importJwk(size_t lengthBits, CryptoAlgorithmIdentifier hash, JsonWebKey&&, bool extractable, CryptoKeyUsageBitmap, CheckAlgCallback&&);

    CryptoKeyClass keyClass() const final { return CryptoKeyClass::HMAC; }

    const Vector<uint8_t>& key() const { return m_key; }
    JsonWebKey exportJwk() const;

    CryptoAlgorithmIdentifier hashAlgorithmIdentifier() const { return m_hash; }

    static ExceptionOr<std::optional<size_t>> getKeyLength(const CryptoAlgorithmParameters&);

private:
    CryptoKeyHMAC(const Vector<uint8_t>& key, CryptoAlgorithmIdentifier hash, bool extractable, CryptoKeyUsageBitmap);
    CryptoKeyHMAC(Vector<uint8_t>&& key, CryptoAlgorithmIdentifier hash, bool extractable, CryptoKeyUsageBitmap);

    KeyAlgorithm algorithm() const final;
    CryptoKey::Data data() const final;

    CryptoAlgorithmIdentifier m_hash;
    Vector<uint8_t> m_key;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CRYPTO_KEY(CryptoKeyHMAC, CryptoKeyClass::HMAC)
