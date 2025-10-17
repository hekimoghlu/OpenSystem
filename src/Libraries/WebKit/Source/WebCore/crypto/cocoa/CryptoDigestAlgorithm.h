/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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

#include <pal/crypto/CryptoDigest.h>

#include "CryptoAlgorithm.h"

namespace WebCore {

static std::optional<PAL::CryptoDigest::Algorithm> cryptoDigestAlgorithm(CryptoAlgorithmIdentifier hashFunction)
{
    switch (hashFunction) {
    case CryptoAlgorithmIdentifier::SHA_1:
        return PAL::CryptoDigest::Algorithm::SHA_1;
    case CryptoAlgorithmIdentifier::SHA_224:
        return PAL::CryptoDigest::Algorithm::SHA_224;
    case CryptoAlgorithmIdentifier::SHA_256:
        return PAL::CryptoDigest::Algorithm::SHA_256;
    case CryptoAlgorithmIdentifier::SHA_384:
        return PAL::CryptoDigest::Algorithm::SHA_384;
    case CryptoAlgorithmIdentifier::SHA_512:
        return PAL::CryptoDigest::Algorithm::SHA_512;
    default:
        return std::nullopt;
    }
}

} // namespace WebCore
