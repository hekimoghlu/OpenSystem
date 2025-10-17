/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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
#include "CryptoAlgorithmAESGCM.h"

#include "CryptoAlgorithmAesGcmParams.h"
#include "CryptoKeyAES.h"
#include "OpenSSLCryptoUniquePtr.h"
#include <openssl/evp.h>

namespace WebCore {

static const EVP_CIPHER* aesAlgorithm(size_t keySize)
{
    if (keySize * 8 == 128)
        return EVP_aes_128_gcm();

    if (keySize * 8 == 192)
        return EVP_aes_192_gcm();

    if (keySize * 8 == 256)
        return EVP_aes_256_gcm();

    return nullptr;
}

static std::optional<Vector<uint8_t>> cryptEncrypt(const Vector<uint8_t>& key, const Vector<uint8_t>& iv, const Vector<uint8_t>& plainText, const Vector<uint8_t>& additionalData, uint8_t tagLength)
{
    const EVP_CIPHER* algorithm = aesAlgorithm(key.size());
    if (!algorithm)
        return std::nullopt;

    EvpCipherCtxPtr ctx;
    int len;

    Vector<uint8_t> cipherText(plainText.size() + tagLength);
    size_t tagOffset = plainText.size();

    // Create and initialize the context
    if (!(ctx = EvpCipherCtxPtr(EVP_CIPHER_CTX_new())))
        return std::nullopt;

    // Disable padding
    if (1 != EVP_CIPHER_CTX_set_padding(ctx.get(), 0))
        return std::nullopt;

    // Initialize the encryption operation
    if (1 != EVP_EncryptInit_ex(ctx.get(), algorithm, nullptr, nullptr, nullptr))
        return std::nullopt;

    // Set IV length
    if (1 != EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, iv.size(), nullptr))
        return std::nullopt;

    // Initialize key and IV
    if (1 != EVP_EncryptInit_ex(ctx.get(), nullptr, nullptr, key.data(), iv.data()))
        return std::nullopt;

    // Provide any AAD data
    if (additionalData.size() > 0) {
        if (1 != EVP_EncryptUpdate(ctx.get(), nullptr, &len, additionalData.data(), additionalData.size()))
            return std::nullopt;
    }

    // Provide the message to be encrypted, and obtain the encrypted output
    if (1 != EVP_EncryptUpdate(ctx.get(), cipherText.data(), &len, plainText.data(), plainText.size()))
        return std::nullopt;

    // Finalize the encryption. Normally ciphertext bytes may be written at
    // this stage, but this does not occur in GCM mode
    if (1 != EVP_EncryptFinal_ex(ctx.get(), cipherText.data() + len, &len))
        return std::nullopt;

    // Get the tag
    if (1 != EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_GET_TAG, tagLength, cipherText.data() + tagOffset))
        return std::nullopt;

    return cipherText;
}

static std::optional<Vector<uint8_t>> cryptDecrypt(const Vector<uint8_t>& key, const Vector<uint8_t>& iv, const Vector<uint8_t>& cipherText, const Vector<uint8_t>& additionalData, uint8_t tagLength)
{
    const EVP_CIPHER* algorithm = aesAlgorithm(key.size());
    if (!algorithm)
        return std::nullopt;

    EvpCipherCtxPtr ctx;
    int len;
    int plainTextLen;
    int cipherTextLen = cipherText.size() - tagLength;

    Vector<uint8_t> plainText(cipherText.size());
    auto tag = cipherText.subvector(cipherTextLen, tagLength);

    // Create and initialize the context
    if (!(ctx = EvpCipherCtxPtr(EVP_CIPHER_CTX_new())))
        return std::nullopt;

    // Disable padding
    if (1 != EVP_CIPHER_CTX_set_padding(ctx.get(), 0))
        return std::nullopt;

    // Initialize the encryption operation
    if (1 != EVP_DecryptInit_ex(ctx.get(), algorithm, nullptr, nullptr, nullptr))
        return std::nullopt;

    // Set IV length
    if (1 != EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_IVLEN, iv.size(), nullptr))
        return std::nullopt;

    // Initialize key and IV
    if (1 != EVP_DecryptInit_ex(ctx.get(), nullptr, nullptr, key.data(), iv.data()))
        return std::nullopt;

    // Provide any AAD data
    if (additionalData.size() > 0) {
        if (1 != EVP_DecryptUpdate(ctx.get(), nullptr, &len, additionalData.data(), additionalData.size()))
            return std::nullopt;
    }

    // Set expected tag value
    if (1 != EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_GCM_SET_TAG, tag.size(), tag.data()))
        return std::nullopt;

    // Provide the message to be encrypted, and obtain the encrypted output
    if (1 != EVP_DecryptUpdate(ctx.get(), plainText.data(), &len, cipherText.data(), cipherTextLen))
        return std::nullopt;
    plainTextLen = len;

    // Finalize the decryption
    if (1 != EVP_DecryptFinal_ex(ctx.get(), plainText.data() + len, &len))
        return std::nullopt;

    plainTextLen += len;

    plainText.shrink(plainTextLen);

    return plainText;
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESGCM::platformEncrypt(const CryptoAlgorithmAesGcmParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& plainText)
{
    auto output = cryptEncrypt(key.key(), parameters.ivVector(), plainText, parameters.additionalDataVector(), parameters.tagLength.value_or(0) / 8);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESGCM::platformDecrypt(const CryptoAlgorithmAesGcmParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& cipherText)
{
    auto output = cryptDecrypt(key.key(), parameters.ivVector(), cipherText, parameters.additionalDataVector(), parameters.tagLength.value_or(0) / 8);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

} // namespace WebCore
