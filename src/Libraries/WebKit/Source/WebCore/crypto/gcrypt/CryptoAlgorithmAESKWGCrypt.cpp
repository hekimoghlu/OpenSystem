/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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
#include "CryptoAlgorithmAESKW.h"

#include "CryptoKeyAES.h"
#include <pal/crypto/gcrypt/Handle.h>
#include <pal/crypto/gcrypt/Utilities.h>

namespace WebCore {

static std::optional<Vector<uint8_t>> gcryptWrapKey(const Vector<uint8_t>& key, const Vector<uint8_t>& data)
{
    // Determine the AES algorithm for the given key size.
    auto algorithm = PAL::GCrypt::aesAlgorithmForKeySize(key.size() * 8);
    if (!algorithm)
        return std::nullopt;

    // Create a new GCrypt cipher object for the AES algorithm and the AES-Wrap cipher mode.
    PAL::GCrypt::Handle<gcry_cipher_hd_t> handle;
    gcry_error_t error = gcry_cipher_open(&handle, *algorithm, GCRY_CIPHER_MODE_AESWRAP, 0);
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

    // Finalize the cipher object before performing the encryption.
    error = gcry_cipher_final(handle);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Perform the encryption. The provided output buffer must be 64 bits larger than the input buffer.
    Vector<uint8_t> output(data.size() + 8);
    error = gcry_cipher_encrypt(handle, output.data(), output.size(), data.data(), data.size());
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    return output;
}

static std::optional<Vector<uint8_t>> gcryptUnwrapKey(const Vector<uint8_t>& key, const Vector<uint8_t>& data)
{
    // Determine the AES algorithm for the given key size.
    auto algorithm = PAL::GCrypt::aesAlgorithmForKeySize(key.size() * 8);
    if (!algorithm)
        return std::nullopt;

    // Create a new GCrypt cipher object for the AES algorithm and the AES-Wrap cipher mode.
    PAL::GCrypt::Handle<gcry_cipher_hd_t> handle;
    gcry_error_t error = gcry_cipher_open(&handle, *algorithm, GCRY_CIPHER_MODE_AESWRAP, 0);
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

    // Finalize the cipher object before performing the encryption.
    error = gcry_cipher_final(handle);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Perform the decryption. The output buffer may be specified 64 bits shorter than the input buffer.
    Vector<uint8_t> output(data.size() - 8);
    error = gcry_cipher_decrypt(handle, output.data(), output.size(), data.data(), data.size());
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    return output;
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESKW::platformWrapKey(const CryptoKeyAES& key, const Vector<uint8_t>& data)
{
    auto output = gcryptWrapKey(key.key(), data);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmAESKW::platformUnwrapKey(const CryptoKeyAES& key, const Vector<uint8_t>& data)
{
    auto output = gcryptUnwrapKey(key.key(), data);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

} // namespace WebCore
