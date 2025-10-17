/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#include "CryptoAlgorithmPBKDF2.h"

#include "CryptoAlgorithmPbkdf2Params.h"
#include "CryptoKeyRaw.h"
#include "GCryptUtilities.h"

namespace WebCore {

static std::optional<Vector<uint8_t>> gcryptDeriveBits(const Vector<uint8_t>& keyData, const Vector<uint8_t>& saltData, CryptoAlgorithmIdentifier hashIdentifier, size_t iterations, size_t length)
{
    // Determine the hash algorithm that will be used for derivation.
    auto hashAlgorithm = digestAlgorithm(hashIdentifier);
    if (!hashAlgorithm)
        return std::nullopt;

    // Length, in bits, is a multiple of 8, as guaranteed by CryptoAlgorithmPBKDF2::deriveBits().
    ASSERT(!(length % 8));

    // Derive bits using PBKDF2, the specified hash algorithm, salt data and iteration count.
    Vector<uint8_t> result(length / 8);
    gcry_error_t error = gcry_kdf_derive(keyData.data(), keyData.size(), GCRY_KDF_PBKDF2, *hashAlgorithm, saltData.data(), saltData.size(), iterations, result.size(), result.data());
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    return result;
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmPBKDF2::platformDeriveBits(const CryptoAlgorithmPbkdf2Params& parameters, const CryptoKeyRaw& key, size_t length)
{
    auto output = gcryptDeriveBits(key.key(), parameters.saltVector(), parameters.hashIdentifier, parameters.iterations, length);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

} // namespace WebCore
