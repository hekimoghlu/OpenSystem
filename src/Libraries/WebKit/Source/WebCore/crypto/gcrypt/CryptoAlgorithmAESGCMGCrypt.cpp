/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#include "NotImplemented.h"
#include <pal/crypto/gcrypt/Handle.h>
#include <pal/crypto/gcrypt/Utilities.h>
#include <wtf/CryptographicUtilities.h>

namespace WebCore {

static std::optional<Vector<uint8_t>> gcryptEncrypt(const Vector<uint8_t>& key, const Vector<uint8_t>& iv, const Vector<uint8_t>& plainText, const Vector<uint8_t>& additionalData, uint8_t tagLength)
{
    // Determine the AES algorithm for the given key size.
    auto algorithm = PAL::GCrypt::aesAlgorithmForKeySize(key.size() * 8);
    if (!algorithm)
        return std::nullopt;

    // Create a new GCrypt cipher object for the AES algorithm and the GCM cipher mode.
    PAL::GCrypt::Handle<gcry_cipher_hd_t> handle;
    gcry_error_t error = gcry_cipher_open(&handle, *algorithm, GCRY_CIPHER_MODE_GCM, GCRY_CIPHER_SECURE);
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

    // Use the given additonal data, if any, as the authentication data for this cipher object.
    if (!additionalData.isEmpty()) {
        error = gcry_cipher_authenticate(handle, additionalData.data(), additionalData.size());
        if (error != GPG_ERR_NO_ERROR) {
            PAL::GCrypt::logError(error);
            return std::nullopt;
        }
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

    // If tag length was specified, retrieve the tag data and append it to the output vector.
    if (tagLength) {
        Vector<uint8_t> tag(tagLength);
        error = gcry_cipher_gettag(handle, tag.data(), tag.size());
        if (error != GPG_ERR_NO_ERROR) {
            PAL::GCrypt::logError(error);
            return std::nullopt;
        }

        output.appendVector(tag);
    }

    return output;
}

static std::optional<Vector<uint8_t>> gcryptDecrypt(const Vector<uint8_t>& key, const Vector<uint8_t>& iv, const Vector<uint8_t>& cipherText, const Vector<uint8_t>& additionalData, uint8_t tagLength)
{
    // Determine the AES algorithm for the given key size.
    auto algorithm = PAL::GCrypt::aesAlgorithmForKeySize(key.size() * 8);
    if (!algorithm)
        return std::nullopt;

    // Create a new GCrypt cipher object for the AES algorithm and the GCM cipher mode.
    PAL::GCrypt::Handle<gcry_cipher_hd_t> handle;
    gcry_error_t error = gcry_cipher_open(&handle, *algorithm, GCRY_CIPHER_MODE_GCM, 0);
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

    // Use the given additonal data, if any, as the authentication data for this cipher object.
    if (!additionalData.isEmpty()) {
        error = gcry_cipher_authenticate(handle, additionalData.data(), additionalData.size());
        if (error != GPG_ERR_NO_ERROR) {
            PAL::GCrypt::logError(error);
            return std::nullopt;
        }
    }

    // Finalize the cipher object before performing the encryption.
    error = gcry_cipher_final(handle);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Account for the specified tag length when performing the decryption and retrieving the decrypted output.
    size_t cipherLength = cipherText.size() - tagLength;
    Vector<uint8_t> output(cipherLength);
    error = gcry_cipher_decrypt(handle, output.data(), output.size(), cipherText.data(), cipherLength);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // If tag length was indeed specified, retrieve the tag data and compare it securely to the tag data that
    // is in the passed-in cipher text Vector, bailing if there is a mismatch and returning the decrypted
    // plaintext otherwise.
    if (tagLength) {
        Vector<uint8_t> tag(tagLength);
        error = gcry_cipher_gettag(handle, tag.data(), tagLength);
        if (error != GPG_ERR_NO_ERROR) {
            PAL::GCrypt::logError(error);
            return std::nullopt;
        }

        if (constantTimeMemcmp(tag.span(), cipherText.subspan(cipherLength)))
            return std::nullopt;
    }

    return output;
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESGCM::platformEncrypt(const CryptoAlgorithmAesGcmParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& plainText)
{
    auto output = gcryptEncrypt(key.key(), parameters.ivVector(), plainText, parameters.additionalDataVector(), parameters.tagLength.value_or(0) / 8);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESGCM::platformDecrypt(const CryptoAlgorithmAesGcmParams& parameters, const CryptoKeyAES& key, const Vector<uint8_t>& cipherText)
{
    auto output = gcryptDecrypt(key.key(), parameters.ivVector(), cipherText, parameters.additionalDataVector(), parameters.tagLength.value_or(0) / 8);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

} // namespace WebCore
