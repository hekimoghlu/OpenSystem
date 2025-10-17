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
#include "config.h"
#include "CryptoAlgorithmAESCBC.h"

#include "CryptoAlgorithmAesCbcCfbParams.h"
#include "CryptoKeyAES.h"
#include "OpenSSLCryptoUniquePtr.h"
#include <openssl/evp.h>

namespace WebCore {

static const EVP_CIPHER* aesAlgorithm(size_t keySize)
{
    if (keySize * 8 == 128)
        return EVP_aes_128_cbc();

    if (keySize * 8 == 192)
        return EVP_aes_192_cbc();

    if (keySize * 8 == 256)
        return EVP_aes_256_cbc();

    return nullptr;
}

static std::optional<Vector<uint8_t>> cryptEncrypt(const Vector<uint8_t>& key, const Vector<uint8_t>& iv, Vector<uint8_t>&& plainText)
{
    const EVP_CIPHER* algorithm = aesAlgorithm(key.size());
    if (!algorithm)
        return std::nullopt;

    EvpCipherCtxPtr ctx;
    int len;

    // Create and initialize the context
    if (!(ctx = EvpCipherCtxPtr(EVP_CIPHER_CTX_new())))
        return std::nullopt;

    size_t plainSize = plainText.size();
    const size_t cipherTextLen = roundUpToMultipleOf(EVP_CIPHER_block_size(algorithm), plainSize + 1);
    Vector<uint8_t> cipherText(cipherTextLen);

    // Initialize the encryption operation
    if (1 != EVP_EncryptInit_ex(ctx.get(), algorithm, nullptr, key.data(), iv.data()))
        return std::nullopt;

    // Provide the message to be encrypted, and obtain the encrypted output
    if (1 != EVP_EncryptUpdate(ctx.get(), cipherText.data(), &len, plainText.data(), plainSize))
        return std::nullopt;

    // Finalize the encryption. Further ciphertext bytes may be written at this stage
    if (1 != EVP_EncryptFinal_ex(ctx.get(), cipherText.data() + len, &len))
        return std::nullopt;

    return cipherText;
}

static std::optional<Vector<uint8_t>> cryptDecrypt(const Vector<uint8_t>& key, const Vector<uint8_t>& iv, const Vector<uint8_t>& cipherText)
{
    const EVP_CIPHER* algorithm = aesAlgorithm(key.size());
    if (!algorithm)
        return std::nullopt;

    EvpCipherCtxPtr ctx;

    size_t cipherSize = cipherText.size();
    Vector<uint8_t> plainText(cipherSize);
    int len;
    int plainTextLen;

    // Create and initialize the context
    if (!(ctx = EvpCipherCtxPtr(EVP_CIPHER_CTX_new())))
        return std::nullopt;

    // Initialize the decryption operation
    if (1 != EVP_DecryptInit_ex(ctx.get(), algorithm, nullptr, key.data(), iv.data()))
        return std::nullopt;

    // Provide the message to be decrypted, and obtain the plaintext output
    if (1 != EVP_DecryptUpdate(ctx.get(), plainText.data(), &len, cipherText.data(), cipherSize))
        return std::nullopt;
    plainTextLen = len;

    // Finalize the decryption. Further plaintext bytes may be written at this stage
    if (1 != EVP_DecryptFinal_ex(ctx.get(), plainText.data() + len, &len))
        return std::nullopt;
    plainTextLen += len;

    plainText.shrink(plainTextLen);

    return plainText;
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESCBC::platformEncrypt(const CryptoAlgorithmAesCbcCfbParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& plainText, Padding)
{
    auto output = cryptEncrypt(key.key(), parameters.ivVector(), Vector<uint8_t>(plainText));
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESCBC::platformDecrypt(const CryptoAlgorithmAesCbcCfbParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& cipherText, Padding)
{
    auto output = cryptDecrypt(key.key(), parameters.ivVector(), cipherText);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

} // namespace WebCore
