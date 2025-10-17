/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
#pragma once

#if HAVE(SWIFT_CPP_INTEROP)
#include "PALSwift.h"

namespace WebCore {
inline PAL::HashFunction toCKHashFunction(CryptoAlgorithmIdentifier hash)
{
    switch (hash) {
    case CryptoAlgorithmIdentifier::SHA_256:
        return PAL::HashFunction::sha256();
        break;
    case CryptoAlgorithmIdentifier::SHA_384:
        return PAL::HashFunction::sha384();
        break;
    case CryptoAlgorithmIdentifier::SHA_512:
        return PAL::HashFunction::sha512();
        break;
    case CryptoAlgorithmIdentifier::SHA_1:
        return PAL::HashFunction::sha1();
        break;
    default:
        break;
    }
    ASSERT_NOT_REACHED();
    return PAL::HashFunction::sha512();
}

inline bool isValidHashParameter(CryptoAlgorithmIdentifier hash)
{
    return hash == CryptoAlgorithmIdentifier::SHA_1 || hash == CryptoAlgorithmIdentifier::SHA_256 || hash == CryptoAlgorithmIdentifier::SHA_512 || hash == CryptoAlgorithmIdentifier::SHA_384;
}

} // WebCore
#endif
