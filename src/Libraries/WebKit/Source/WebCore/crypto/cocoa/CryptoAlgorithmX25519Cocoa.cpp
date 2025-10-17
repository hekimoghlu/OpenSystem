/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
#include "CryptoAlgorithmX25519.h"

#include "CryptoKeyOKP.h"
#if HAVE(SWIFT_CPP_INTEROP)
#include <pal/PALSwift.h>
#endif
#include <pal/spi/cocoa/CoreCryptoSPI.h>

namespace WebCore {

#if HAVE(SWIFT_CPP_INTEROP)
static std::optional<Vector<uint8_t>> deriveBitsCryptoKit(const Vector<uint8_t>& baseKey, const Vector<uint8_t>& publicKey)
{
    if (baseKey.size() != ed25519KeySize || publicKey.size() != ed25519KeySize)
        return std::nullopt;
    auto rv = PAL::EdKey::deriveBits(PAL::EdKeyAgreementAlgorithm::x25519(), baseKey.span(), publicKey.span());
    if (rv.errorCode != Cpp::ErrorCodes::Success)
        return std::nullopt;
    return WTFMove(rv.result);
}
#else
static std::optional<Vector<uint8_t>> deriveBitsCoreCrypto(const Vector<uint8_t>& baseKey, const Vector<uint8_t>& publicKey)
{
    if (baseKey.size() != ed25519KeySize || publicKey.size() != ed25519KeySize)
        return std::nullopt;

    ccec25519pubkey derivedKey;
    static_assert(sizeof(derivedKey) == ed25519KeySize);
#if HAVE(CORE_CRYPTO_SIGNATURES_INT_RETURN_VALUE)
    if (cccurve25519(derivedKey, baseKey.data(), publicKey.data()))
        return std::nullopt;
#else
    cccurve25519(derivedKey, baseKey.data(), publicKey.data());
#endif
    return Vector<uint8_t>(std::span { derivedKey });
}
#endif
std::optional<Vector<uint8_t>> CryptoAlgorithmX25519::platformDeriveBits(const CryptoKeyOKP& baseKey, const CryptoKeyOKP& publicKey)
{
#if HAVE(SWIFT_CPP_INTEROP)
    return deriveBitsCryptoKit(baseKey.platformKey(), publicKey.platformKey());
#else
    return deriveBitsCoreCrypto(baseKey.platformKey(), publicKey.platformKey());
#endif
}
} // namespace WebCore
