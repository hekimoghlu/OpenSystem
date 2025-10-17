/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#include "CryptoAlgorithmHKDF.h"

#include "CommonCryptoUtilities.h"
#include "CryptoAlgorithmHkdfParams.h"
#include "CryptoKeyRaw.h"
#include "CryptoUtilitiesCocoa.h"
#if HAVE(SWIFT_CPP_INTEROP)
#include <pal/PALSwiftUtils.h>
#endif

namespace WebCore {

static ExceptionOr<Vector<uint8_t>> platformDeriveBitsCC(const CryptoAlgorithmHkdfParams& parameters, const CryptoKeyRaw& key, size_t length)
{
    CCDigestAlgorithm digestAlgorithm;
    getCommonCryptoDigestAlgorithm(parameters.hashIdentifier, digestAlgorithm);

    return deriveHDKFBits(digestAlgorithm, key.key().span(), parameters.saltVector().span(), parameters.infoVector().span(), length);
}

#if HAVE(SWIFT_CPP_INTEROP)
static ExceptionOr<Vector<uint8_t>> platformDeriveBitsCryptoKit(const CryptoAlgorithmHkdfParams& parameters, const CryptoKeyRaw& key, size_t length)
{
    if (!isValidHashParameter(parameters.hashIdentifier))
        return Exception { ExceptionCode::OperationError };
    auto rv = PAL::HKDF::deriveBits(key.key().span(), parameters.saltVector().span(), parameters.infoVector().span(), length, toCKHashFunction(parameters.hashIdentifier));
    if (rv.errorCode != Cpp::ErrorCodes::Success)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(rv.result);
}
#endif

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmHKDF::platformDeriveBits(const CryptoAlgorithmHkdfParams& parameters, const CryptoKeyRaw& key, size_t length)
{
#if HAVE(SWIFT_CPP_INTEROP)
    if (parameters.hashIdentifier != CryptoAlgorithmIdentifier::SHA_224)
        return platformDeriveBitsCryptoKit(parameters, key, length);
    return platformDeriveBitsCC(parameters, key, length);
#else
    return platformDeriveBitsCC(parameters, key, length);
#endif
}
} // namespace WebCore
