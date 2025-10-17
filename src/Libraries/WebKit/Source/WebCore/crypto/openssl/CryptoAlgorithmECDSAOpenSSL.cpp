/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
#include "CryptoAlgorithmECDSA.h"

#include "CryptoAlgorithmEcdsaParams.h"
#include "CryptoKeyEC.h"
#include "OpenSSLUtilities.h"

namespace WebCore {

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmECDSA::platformSign(const CryptoAlgorithmEcdsaParams& parameters, const CryptoKeyEC& key, const Vector<uint8_t>& data)
{
    size_t keySizeInBytes = (key.keySizeInBits() + 7) / 8;

    const EVP_MD* md = digestAlgorithm(parameters.hashIdentifier);
    if (!md)
        return Exception { ExceptionCode::NotSupportedError };

    std::optional<Vector<uint8_t>> digest = calculateDigest(md, data);
    if (!digest)
        return Exception { ExceptionCode::OperationError };

    EC_KEY* ecKey = EVP_PKEY_get0_EC_KEY(key.platformKey().get());
    if (!ecKey)
        return Exception { ExceptionCode::OperationError };

    // We use ECDSA_do_sign rather than EVP API because the latter handles ECDSA signature in DER format
    // while this function is supposed to return simply concatinated "r" and "s".
    auto sig = ECDSASigPtr(ECDSA_do_sign(digest->data(), digest->size(), ecKey));
    if (!sig)
        return Exception { ExceptionCode::OperationError };

    const BIGNUM* r;
    const BIGNUM* s;
    ECDSA_SIG_get0(sig.get(), &r, &s);

    // Concatenate r and s, expanding r and s to keySizeInBytes.
    Vector<uint8_t> signature = convertToBytesExpand(r, keySizeInBytes);
    signature.appendVector(convertToBytesExpand(s, keySizeInBytes));

    return signature;
}

ExceptionOr<bool> CryptoAlgorithmECDSA::platformVerify(const CryptoAlgorithmEcdsaParams& parameters, const CryptoKeyEC& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
    size_t keySizeInBytes = (key.keySizeInBits() + 7) / 8;

    // Bail if the signature size isn't double the key size (i.e. concatenated r and s components).
    if (signature.size() != keySizeInBytes * 2)
        return false;
    
    auto sig = ECDSASigPtr(ECDSA_SIG_new());
    auto r = BN_bin2bn(signature.data(), keySizeInBytes, nullptr);
    auto s = BN_bin2bn(signature.data() + keySizeInBytes, keySizeInBytes, nullptr);

    if (!ECDSA_SIG_set0(sig.get(), r, s))
        return Exception { ExceptionCode::OperationError };

    const EVP_MD* md = digestAlgorithm(parameters.hashIdentifier);
    if (!md)
        return Exception { ExceptionCode::NotSupportedError };

    std::optional<Vector<uint8_t>> digest = calculateDigest(md, data);
    if (!digest)
        return Exception { ExceptionCode::OperationError };

    EC_KEY* ecKey = EVP_PKEY_get0_EC_KEY(key.platformKey().get());
    if (!ecKey)
        return Exception { ExceptionCode::OperationError };

    int ret = ECDSA_do_verify(digest->data(), digest->size(), sig.get(), ecKey);
    return ret == 1;
}

} // namespace WebCore
