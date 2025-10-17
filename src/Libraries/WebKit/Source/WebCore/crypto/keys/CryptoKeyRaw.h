/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 16, 2023.
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

namespace WebCore {

class CryptoKeyRaw final : public CryptoKey {
public:
    static Ref<CryptoKeyRaw> create(CryptoAlgorithmIdentifier identifier, Vector<uint8_t>&& keyData, CryptoKeyUsageBitmap usages)
    {
        return adoptRef(*new CryptoKeyRaw(identifier, WTFMove(keyData), usages));
    }

    const Vector<uint8_t>& key() const { return m_key; }

private:
    CryptoKeyRaw(CryptoAlgorithmIdentifier, Vector<uint8_t>&& keyData, CryptoKeyUsageBitmap);

    CryptoKeyClass keyClass() const final { return CryptoKeyClass::Raw; }
    KeyAlgorithm algorithm() const final;
    CryptoKey::Data data() const final;

    Vector<uint8_t> m_key;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CRYPTO_KEY(CryptoKeyRaw, CryptoKeyClass::Raw)
