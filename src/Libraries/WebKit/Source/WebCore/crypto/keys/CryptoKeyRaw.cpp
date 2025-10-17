/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
#include "CryptoKeyRaw.h"

#include "CryptoAlgorithmRegistry.h"

namespace WebCore {

CryptoKeyRaw::CryptoKeyRaw(CryptoAlgorithmIdentifier identifier, Vector<uint8_t>&& keyData, CryptoKeyUsageBitmap usages)
    : CryptoKey(identifier, CryptoKeyType::Secret, false, usages)
    , m_key(WTFMove(keyData))
{
}

auto CryptoKeyRaw::algorithm() const -> KeyAlgorithm
{
    CryptoKeyAlgorithm result;
    result.name = CryptoAlgorithmRegistry::singleton().name(algorithmIdentifier());
    return result;
}

CryptoKey::Data CryptoKeyRaw::data() const
{
    return CryptoKey::Data {
        CryptoKeyClass::Raw,
        algorithmIdentifier(),
        extractable(),
        usagesBitmap(),
        { key() },
    };
}

} // namespace WebCore
