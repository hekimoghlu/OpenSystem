/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#include "CryptoAlgorithmAESCBC.h"

#include "CryptoAlgorithmAesCbcCfbParams.h"
#include "CryptoKeyAES.h"
#include <CommonCrypto/CommonCrypto.h>
#include <wtf/text/ParsingUtilities.h>

namespace WebCore {

static ExceptionOr<Vector<uint8_t>> transformAESCBC(CCOperation operation, const Vector<uint8_t>& iv, const Vector<uint8_t>& key, const Vector<uint8_t>& data, CryptoAlgorithmAESCBC::Padding padding)
{
    CCOptions options = padding == CryptoAlgorithmAESCBC::Padding::Yes ? kCCOptionPKCS7Padding : 0;
    CCCryptorRef cryptor;
    CCCryptorStatus status = CCCryptorCreate(operation, kCCAlgorithmAES, options, key.data(), key.size(), iv.data(), &cryptor);
    if (status)
        return Exception { ExceptionCode::OperationError };

    Vector<uint8_t> result(CCCryptorGetOutputLength(cryptor, data.size(), true));

    size_t bytesWritten;
    status = CCCryptorUpdate(cryptor, data.data(), data.size(), result.data(), result.size(), &bytesWritten);
    if (status)
        return Exception { ExceptionCode::OperationError };

    auto p = result.mutableSpan().subspan(bytesWritten);
    if (padding == CryptoAlgorithmAESCBC::Padding::Yes) {
        status = CCCryptorFinal(cryptor, p.data(), p.size(), &bytesWritten);
        skip(p, bytesWritten);
        if (status)
            return Exception { ExceptionCode::OperationError };
    }

    result.shrink(result.size() - p.size());

    CCCryptorRelease(cryptor);

    return WTFMove(result);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESCBC::platformEncrypt(const CryptoAlgorithmAesCbcCfbParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& plainText, Padding padding)
{
    ASSERT(parameters.ivVector().size() == kCCBlockSizeAES128 || parameters.ivVector().isEmpty());
    ASSERT(padding == Padding::Yes || !(plainText.size() % kCCBlockSizeAES128));
    return transformAESCBC(kCCEncrypt, parameters.ivVector(), key.key(), plainText, padding);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESCBC::platformDecrypt(const CryptoAlgorithmAesCbcCfbParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& cipherText, Padding padding)
{
    ASSERT(parameters.ivVector().size() == kCCBlockSizeAES128 || parameters.ivVector().isEmpty());
    ASSERT(padding == Padding::Yes || !(cipherText.size() % kCCBlockSizeAES128));
    auto result = transformAESCBC(kCCDecrypt, parameters.ivVector(), key.key(), cipherText, Padding::No);
    if (result.hasException())
        return result.releaseException();

    auto data = result.releaseReturnValue();
    if (padding == Padding::Yes && !data.isEmpty()) {
        // Validate and remove padding as per https://www.w3.org/TR/WebCryptoAPI/#aes-cbc-operations (Decrypt section).
        auto paddingByte = data.last();
        if (!paddingByte || paddingByte > 16 || paddingByte > data.size())
            return Exception { ExceptionCode::OperationError };
        for (size_t i = data.size() - paddingByte; i < data.size() - 1; ++i) {
            if (data[i] != paddingByte)
                return Exception { ExceptionCode::OperationError };
        }
        data.shrink(data.size() - paddingByte);
    }
    return data;
}

} // namespace WebCore
