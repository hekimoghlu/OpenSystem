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
#include <CommonCrypto/CommonKeyDerivation.h>

namespace WebCore {

namespace CryptoAlgorithmPBKDF2MacInternal {
static CCPseudoRandomAlgorithm commonCryptoHMACAlgorithm(CryptoAlgorithmIdentifier hashFunction)
{
    switch (hashFunction) {
    case CryptoAlgorithmIdentifier::SHA_1:
        return kCCPRFHmacAlgSHA1;
    case CryptoAlgorithmIdentifier::SHA_224:
        return kCCPRFHmacAlgSHA224;
    case CryptoAlgorithmIdentifier::SHA_256:
        return kCCPRFHmacAlgSHA256;
    case CryptoAlgorithmIdentifier::SHA_384:
        return kCCPRFHmacAlgSHA384;
    case CryptoAlgorithmIdentifier::SHA_512:
        return kCCPRFHmacAlgSHA512;
    default:
        ASSERT_NOT_REACHED();
        return 0;
    }
}
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmPBKDF2::platformDeriveBits(const CryptoAlgorithmPbkdf2Params& parameters, const CryptoKeyRaw& key, size_t length)
{
    Vector<uint8_t> result(length / 8);
    // CCKeyDerivationPBKDF returns an error if the key pointer is null, even if the key length is 0. For this reason, we need to make sure we pass the empty string
    // instead of a null pointer in this case.
    if (CCKeyDerivationPBKDF(kCCPBKDF2, key.key().data() ? reinterpret_cast<const char *>(key.key().data()) : "", key.key().size(), parameters.saltVector().data(), parameters.saltVector().size(), CryptoAlgorithmPBKDF2MacInternal::commonCryptoHMACAlgorithm(parameters.hashIdentifier), parameters.iterations, result.data(), length / 8))
        return Exception { ExceptionCode::OperationError };
    return WTFMove(result);
}

} // namespace WebCore
