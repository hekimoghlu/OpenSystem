/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#include "CryptoAlgorithmAESCFB.h"

#include "CryptoAlgorithmAesCbcCfbParams.h"
#include "CryptoKeyAES.h"
#include <CommonCrypto/CommonCrypto.h>

namespace WebCore {

static ExceptionOr<Vector<uint8_t>> transformAESCFB(CCOperation operation, const Vector<uint8_t>& iv, const Vector<uint8_t>& key, const Vector<uint8_t>& data)
{
    CCCryptorRef cryptor;
    CCCryptorStatus status = CCCryptorCreateWithMode(operation, kCCModeCFB8, kCCAlgorithmAES, ccNoPadding, iv.data(), key.data(), key.size(), 0, 0, 0, 0, &cryptor);
    if (status)
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> result(CCCryptorGetOutputLength(cryptor, data.size(), true));

    size_t bytesWritten;
    status = CCCryptorUpdate(cryptor, data.data(), data.size(), result.data(), result.size(), &bytesWritten);
    if (status)
        return Exception { ExceptionCode::OperationError };

    auto p = result.mutableSpan().subspan(bytesWritten);
    status = CCCryptorFinal(cryptor, p.data(), p.size(), &bytesWritten);
    skip(p, bytesWritten);
    if (status)
        return Exception { ExceptionCode::OperationError };

    result.shrink(result.size() - p.size());

    CCCryptorRelease(cryptor);

    return WTFMove(result);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESCFB::platformEncrypt(const CryptoAlgorithmAesCbcCfbParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& plainText)
{
    ASSERT(parameters.ivVector().size() == kCCBlockSizeAES128);
    return transformAESCFB(kCCEncrypt, parameters.ivVector(), key.key(), plainText);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESCFB::platformDecrypt(const CryptoAlgorithmAesCbcCfbParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& cipherText)
{
    ASSERT(parameters.ivVector().size() == kCCBlockSizeAES128);
    return transformAESCFB(kCCDecrypt, parameters.ivVector(), key.key(), cipherText);
}

} // namespace WebCore
