/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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

#include "CommonCryptoUtilities.h"
#include "CryptoKeyEC.h"
#if HAVE(SWIFT_CPP_INTEROP)
#include <pal/PALSwift.h>
#endif

namespace WebCore {

#if !HAVE(SWIFT_CPP_INTEROP)
static std::optional<Vector<uint8_t>> platformDeriveBitsCC(const CryptoKeyEC& baseKey, const CryptoKeyEC& publicKey)
{
    std::optional<Vector<uint8_t>> result = std::nullopt;
    Vector<uint8_t> derivedKey(baseKey.keySizeInBytes()); // Per https://tools.ietf.org/html/rfc6090#section-4.
    size_t size = derivedKey.size();

    if (!CCECCryptorComputeSharedSecret(baseKey.platformKey().get(), publicKey.platformKey().get(), derivedKey.data(), &size))
        result = std::make_optional(WTFMove(derivedKey));
    return result;
}
#else
static std::optional<Vector<uint8_t>> platformDeriveBitsCryptoKit(const CryptoKeyEC& baseKey, const CryptoKeyEC& publicKey)
{
    auto rv = baseKey.platformKey()->deriveBits(publicKey.platformKey());
    if (rv.errorCode != Cpp::ErrorCodes::Success)
        return std::nullopt;
    return std::make_optional(WTFMove(rv.result));
}
#endif

std::optional<Vector<uint8_t>> CryptoAlgorithmECDH::platformDeriveBits(const CryptoKeyEC& baseKey, const CryptoKeyEC& publicKey)
{
#if HAVE(SWIFT_CPP_INTEROP)
    return platformDeriveBitsCryptoKit(baseKey, publicKey);
#else
    return platformDeriveBitsCC(baseKey, publicKey);
#endif
}

} // namespace WebCore
