/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#include "OpenSSLUtilities.h"
#include <openssl/evp.h>

namespace WebCore {

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmPBKDF2::platformDeriveBits(const CryptoAlgorithmPbkdf2Params& parameters, const CryptoKeyRaw& key, size_t length)
{
    auto algorithm = digestAlgorithm(parameters.hashIdentifier);
    if (!algorithm)
        return Exception { ExceptionCode::NotSupportedError };

    // iterations must not be zero.
    if (!parameters.iterations)
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> output(length / 8);
    if (PKCS5_PBKDF2_HMAC(reinterpret_cast<const char*>(key.key().data()), key.key().size(), parameters.saltVector().data(), parameters.saltVector().size(), parameters.iterations, algorithm, output.size(), output.data()) <= 0)
        return Exception { ExceptionCode::OperationError };

    return output;
}

} // namespace WebCore
