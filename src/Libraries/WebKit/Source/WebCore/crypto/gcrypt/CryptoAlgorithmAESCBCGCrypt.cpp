/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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
#include "NotImplemented.h"
#include <pal/crypto/gcrypt/Handle.h>
#include <pal/crypto/gcrypt/Utilities.h>

namespace WebCore {

static std::optional<Vector<uint8_t>> gcryptEncrypt(const Vector<uint8_t>& key, const Vector<uint8_t>& iv, Vector<uint8_t>&& plainText)
{
    // Determine the AES algorithm for the given key size.
    auto algorithm = PAL::GCrypt::aesAlgorithmForKeySize(key.size() * 8);
    if (!algorithm)
        return std::nullopt;

    // Create a new GCrypt cipher object for the AES algorithm and the CBC cipher mode.
    PAL::GCrypt::Handle<gcry_cipher_hd_t> handle;
    gcry_error_t error = gcry_cipher_open(&handle, *algorithm, GCRY_CIPHER_MODE_CBC, 0);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Use the given key for this cipher object.
    error = gcry_cipher_setkey(handle, key.data(), key.size());
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Use the given IV for this cipher object.
    error = gcry_cipher_setiv(handle, iv.data(), iv.size());
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Use the PKCS#7 padding.
    {
        // Round up the size value to the next multiple of the cipher's block length to get the padded size.
        size_t size = plainText.size();
        size_t paddedSize = roundUpToMultipleOf(gcry_cipher_get_algo_blklen(*algorithm), size + 1);

        // Padded size should be bigger than size, but bail if the value doesn't fit into a byte.
        ASSERT(paddedSize > size);
        if (paddedSize - size > 255)
            return std::nullopt;
        uint8_t paddingValue = paddedSize - size;

        plainText.insertFill(size, paddingValue, paddingValue);
    }

    // Finalize the cipher object before performing the encryption.
    error = gcry_cipher_final(handle);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Perform the encryption and retrieve the encrypted output.
    Vector<uint8_t> output(plainText.size());
    error = gcry_cipher_encrypt(handle, output.data(), output.size(), plainText.data(), plainText.size());
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    return output;
}

static std::optional<Vector<uint8_t>> gcryptDecrypt(const Vector<uint8_t>& key, const Vector<uint8_t>& iv, const Vector<uint8_t>& cipherText)
{
    // Determine the AES algorithm for the given key size.
    auto algorithm = PAL::GCrypt::aesAlgorithmForKeySize(key.size() * 8);
    if (!algorithm)
        return std::nullopt;

    // Create a new GCrypt cipher object for the AES algorithm and the CBC cipher mode.
    PAL::GCrypt::Handle<gcry_cipher_hd_t> handle;
    gcry_error_t error = gcry_cipher_open(&handle, *algorithm, GCRY_CIPHER_MODE_CBC, 0);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Use the given key for this cipher object.
    error = gcry_cipher_setkey(handle, key.data(), key.size());
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Use the given IV for this cipher object.
    error = gcry_cipher_setiv(handle, iv.data(), iv.size());
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Finalize the cipher object before performing the decryption.
    error = gcry_cipher_final(handle);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Perform the decryption and retrieve the decrypted output.
    Vector<uint8_t> output(cipherText.size());
    error = gcry_cipher_decrypt(handle, output.data(), output.size(), cipherText.data(), cipherText.size());
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Remove the PKCS#7 padding from the decrypted output.
    {
        // The padding value can be retrieved from the last byte.
        uint8_t paddingValue = output.last();
        if (paddingValue > gcry_cipher_get_algo_blklen(*algorithm))
            return std::nullopt;

        // Padding value mustn't be greater than the size of the padded output.
        size_t size = output.size();
        if (paddingValue > size)
            return std::nullopt;

        // Bail if the last `paddingValue` bytes don't have the value of `paddingValue`.
        auto padding = output.subspan(size - paddingValue, paddingValue);
        if (std::count(padding.begin(), padding.end(), paddingValue) != paddingValue)
            return std::nullopt;

        // Shrink the output Vector object to drop the PKCS#7 padding.
        output.shrink(size - paddingValue);
    }

    return output;
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESCBC::platformEncrypt(const CryptoAlgorithmAesCbcCfbParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& plainText, Padding)
{
    auto output = gcryptEncrypt(key.key(), parameters.ivVector(), Vector<uint8_t>(plainText));
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESCBC::platformDecrypt(const CryptoAlgorithmAesCbcCfbParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& cipherText, Padding)
{
    auto output = gcryptDecrypt(key.key(), parameters.ivVector(), cipherText);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

} // namespace WebCore
