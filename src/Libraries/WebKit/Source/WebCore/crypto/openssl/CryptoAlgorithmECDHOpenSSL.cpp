/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
#include "CryptoAlgorithmECDH.h"

#include "CryptoKeyEC.h"
#include "OpenSSLUtilities.h"

namespace WebCore {

std::optional<Vector<uint8_t>> CryptoAlgorithmECDH::platformDeriveBits(const CryptoKeyEC& baseKey, const CryptoKeyEC& publicKey)
{
    auto ctx = EvpPKeyCtxPtr(EVP_PKEY_CTX_new(baseKey.platformKey().get(), nullptr));
    if (!ctx)
        return std::nullopt;

    if (EVP_PKEY_derive_init(ctx.get()) <= 0)
        return std::nullopt;

    if (EVP_PKEY_derive_set_peer(ctx.get(), publicKey.platformKey().get()) <= 0)
        return std::nullopt;

    // Call with a nullptr to get the required buffer size.
    size_t keyLen;
    if (EVP_PKEY_derive(ctx.get(), nullptr, &keyLen) <= 0)
        return std::nullopt;

    Vector<uint8_t> key(keyLen);
    if (EVP_PKEY_derive(ctx.get(), key.data(), &keyLen) <= 0)
        return std::nullopt;

    // Shrink the buffer since the new keyLen may differ from the buffer size.
    key.shrink(keyLen);

    return key;
}

} // namespace WebCore
