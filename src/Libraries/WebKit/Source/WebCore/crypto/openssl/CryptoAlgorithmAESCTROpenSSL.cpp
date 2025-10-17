/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#include "CryptoAlgorithmAESCTR.h"

#include "CryptoAlgorithmAesCtrParams.h"
#include "CryptoKeyAES.h"
#include "OpenSSLCryptoUniquePtr.h"
#include <openssl/evp.h>

namespace WebCore {

static const EVP_CIPHER* aesAlgorithm(size_t keySize)
{
    if (keySize * 8 == 128)
        return EVP_aes_128_ctr();

    if (keySize * 8 == 192)
        return EVP_aes_192_ctr();

    if (keySize * 8 == 256)
        return EVP_aes_256_ctr();

    return nullptr;
}

static std::optional<Vector<uint8_t>> crypt(int operation, const Vector<uint8_t>& key, const Vector<uint8_t>& counter, size_t counterLength, const Vector<uint8_t>& inputText)
{
    constexpr size_t blockSize = 16;
    const EVP_CIPHER* algorithm = aesAlgorithm(key.size());
    if (!algorithm)
        return std::nullopt;

    EvpCipherCtxPtr ctx;
    int len;

    // Create and initialize the context
    if (!(ctx = EvpCipherCtxPtr(EVP_CIPHER_CTX_new())))
        return std::nullopt;

    const size_t blocks = roundUpToMultipleOf(blockSize, inputText.size()) / blockSize;

    // Detect loop
    if (counterLength < sizeof(size_t) * 8 && blocks > ((size_t)1 << counterLength))
        return std::nullopt;

    // Calculate capacity before overflow
    CryptoAlgorithmAESCTR::CounterBlockHelper counterBlockHelper(counter, counterLength);
    size_t capacity = counterBlockHelper.countToOverflowSaturating();

    // Divide data into two parts if necessary
    size_t headSize = inputText.size();
    if (capacity < blocks)
        headSize = capacity * blockSize;

    Vector<uint8_t> outputText(inputText.size());
    // First part
    {
        // Initialize the encryption(decryption) operation
        if (1 != EVP_CipherInit_ex(ctx.get(), algorithm, nullptr, key.data(), counter.data(), operation))
            return std::nullopt;

        // Disable padding
        if (1 != EVP_CIPHER_CTX_set_padding(ctx.get(), 0))
            return std::nullopt;

        // Provide the message to be encrypted(decrypted), and obtain the encrypted(decrypted) output
        if (1 != EVP_CipherUpdate(ctx.get(), outputText.data(), &len, inputText.data(), headSize))
            return std::nullopt;

        // Finalize the encryption(decryption)
        if (1 != EVP_CipherFinal_ex(ctx.get(), outputText.data() + len, &len))
            return std::nullopt;
    }

    // Sedond part
    if (capacity < blocks) {
        size_t tailSize = inputText.size() - headSize;

        Vector<uint8_t> remainingCounter = counterBlockHelper.counterVectorAfterOverflow();

        // Initialize the encryption(decryption) operation
        if (1 != EVP_CipherInit_ex(ctx.get(), algorithm, nullptr, key.data(), remainingCounter.data(), operation))
            return std::nullopt;

        // Disable padding
        if (1 != EVP_CIPHER_CTX_set_padding(ctx.get(), 0))
            return std::nullopt;

        // Provide the message to be encrypted(decrypted), and obtain the encrypted(decrypted) output
        if (1 != EVP_CipherUpdate(ctx.get(), outputText.data() + headSize, &len, inputText.data() + headSize, tailSize))
            return std::nullopt;

        // Finalize the encryption(decryption)
        if (1 != EVP_CipherFinal_ex(ctx.get(), outputText.data() + headSize + len, &len))
            return std::nullopt;
    }

    return outputText;
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESCTR::platformEncrypt(const CryptoAlgorithmAesCtrParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& plainText)
{
    auto output = crypt(1, key.key(), parameters.counterVector(), parameters.length, plainText);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESCTR::platformDecrypt(const CryptoAlgorithmAesCtrParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& cipherText)
{
    auto output = crypt(0, key.key(), parameters.counterVector(), parameters.length, cipherText);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

} // namespace WebCore
