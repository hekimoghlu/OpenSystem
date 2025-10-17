/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
#include "CryptoAlgorithmEd25519.h"

#include "CryptoKeyOKP.h"
#if HAVE(SWIFT_CPP_INTEROP)
#include <pal/PALSwift.h>
#endif
#include <pal/spi/cocoa/CoreCryptoSPI.h>

namespace WebCore {

#if !HAVE(SWIFT_CPP_INTEROP)
static ExceptionOr<Vector<uint8_t>> signEd25519(const Vector<uint8_t>& sk, const Vector<uint8_t>& data)
{
    if (sk.size() != ed25519KeySize)
        return Exception { ExceptionCode::OperationError };
    ccec25519pubkey pk;
    const struct ccdigest_info* di = ccsha512_di();
    if (cced25519_make_pub(di, pk, sk.data()))
        return Exception { ExceptionCode::OperationError };
    ccec25519signature newSignature;

#if HAVE(CORE_CRYPTO_SIGNATURES_INT_RETURN_VALUE)
    if (cced25519_sign(di, newSignature, data.size(), data.data(), pk, sk.data()))
        return Exception { ExceptionCode::OperationError };
#else
    cced25519_sign(di, newSignature, data.size(), data.data(), pk, sk.data());
#endif
    return Vector<uint8_t> { std::span { newSignature } };
}

static ExceptionOr<bool> verifyEd25519(const Vector<uint8_t>& key, const Vector<uint8_t>& signature, const Vector<uint8_t> data)
{
    if (key.size() != ed25519KeySize || signature.size() != ed25519SignatureSize)
        return false;
    const struct ccdigest_info* di = ccsha512_di();
    return !cced25519_verify(di, data.size(), data.data(), signature.data(), key.data());
}
#else
static ExceptionOr<Vector<uint8_t>> signEd25519CryptoKit(const Vector<uint8_t>&sk, const Vector<uint8_t>& data)
{
    if (sk.size() != ed25519KeySize)
        return Exception { ExceptionCode::OperationError };
    auto rv = PAL::EdKey::sign(PAL::EdSigningAlgorithm::ed25519(), sk.span(), data.span());
    if (rv.errorCode != Cpp::ErrorCodes::Success)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(rv.result);
}

static ExceptionOr<bool>  verifyEd25519CryptoKit(const Vector<uint8_t>& pubKey, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
    if (pubKey.size() != ed25519KeySize || signature.size() != ed25519SignatureSize)
        return false;
    auto rv = PAL::EdKey::verify(PAL::EdSigningAlgorithm::ed25519(), pubKey.span(), signature.span(), data.span());
    return rv.errorCode == Cpp::ErrorCodes::Success;
}
#endif

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmEd25519::platformSign(const CryptoKeyOKP& key, const Vector<uint8_t>& data)
{
#if HAVE(SWIFT_CPP_INTEROP)
    return signEd25519CryptoKit(key.platformKey(), data);
#else
    return signEd25519(key.platformKey(), data);
#endif
}

ExceptionOr<bool> CryptoAlgorithmEd25519::platformVerify(const CryptoKeyOKP& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
#if HAVE(SWIFT_CPP_INTEROP)
    return verifyEd25519CryptoKit(key.platformKey(), signature, data);
#else
    return verifyEd25519(key.platformKey(), signature, data);
#endif
}

} // namespace WebCore
