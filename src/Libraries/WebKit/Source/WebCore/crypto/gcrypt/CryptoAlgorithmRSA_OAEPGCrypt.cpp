/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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
#include "CryptoAlgorithmRSA_OAEP.h"

#include "CryptoAlgorithmRsaOaepParams.h"
#include "CryptoKeyRSA.h"
#include "GCryptUtilities.h"
#include "NotImplemented.h"

namespace WebCore {

static std::optional<Vector<uint8_t>> gcryptEncrypt(CryptoAlgorithmIdentifier hashAlgorithmIdentifier, gcry_sexp_t keySexp, const Vector<uint8_t>& labelVector, const Vector<uint8_t>& plainText, size_t keySizeInBytes)
{
    // Embed the plain-text data in a data s-expression using OAEP padding.
    // Empty label data is properly handled by gcry_sexp_build().
    PAL::GCrypt::Handle<gcry_sexp_t> dataSexp;
    {
        auto shaAlgorithm = hashAlgorithmName(hashAlgorithmIdentifier);
        if (!shaAlgorithm)
            return std::nullopt;

        gcry_error_t error = gcry_sexp_build(&dataSexp, nullptr, "(data(flags oaep)(hash-algo %s)(label %b)(value %b))",
            shaAlgorithm->characters(), labelVector.size(), labelVector.data(), plainText.size(), plainText.data());
        if (error != GPG_ERR_NO_ERROR) {
            PAL::GCrypt::logError(error);
            return std::nullopt;
        }
    }

    // Encrypt data with the provided key. The returned s-expression is of this form:
    // (enc-val
    //   (flags oaep)
    //   (rsa
    //     (a a-mpi)))
    PAL::GCrypt::Handle<gcry_sexp_t> cipherSexp;
    gcry_error_t error = gcry_pk_encrypt(&cipherSexp, dataSexp, keySexp);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Return MPI data of the embedded a integer.
    PAL::GCrypt::Handle<gcry_sexp_t> aSexp(gcry_sexp_find_token(cipherSexp, "a", 0));
    if (!aSexp)
        return std::nullopt;

    return mpiZeroPrefixedData(aSexp, keySizeInBytes);
}

static std::optional<Vector<uint8_t>> gcryptDecrypt(CryptoAlgorithmIdentifier hashAlgorithmIdentifier, gcry_sexp_t keySexp, const Vector<uint8_t>& labelVector, const Vector<uint8_t>& cipherText)
{
    // Embed the cipher-text data in an enc-val s-expression using OAEP padding.
    // Empty label data is properly handled by gcry_sexp_build().
    PAL::GCrypt::Handle<gcry_sexp_t> encValSexp;
    {
        auto shaAlgorithm = hashAlgorithmName(hashAlgorithmIdentifier);
        if (!shaAlgorithm)
            return std::nullopt;

        gcry_error_t error = gcry_sexp_build(&encValSexp, nullptr, "(enc-val(flags oaep)(hash-algo %s)(label %b)(rsa(a %b)))",
            shaAlgorithm->characters(), labelVector.size(), labelVector.data(), cipherText.size(), cipherText.data());
        if (error != GPG_ERR_NO_ERROR) {
            PAL::GCrypt::logError(error);
            return std::nullopt;
        }
    }

    // Decrypt data with the provided key. The returned s-expression is of this form:
    // (data
    //   (flags oaep)
    //   (value block))
    PAL::GCrypt::Handle<gcry_sexp_t> plainSexp;
    gcry_error_t error = gcry_pk_decrypt(&plainSexp, encValSexp, keySexp);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Return MPI data of the embedded value integer.
    PAL::GCrypt::Handle<gcry_sexp_t> valueSexp(gcry_sexp_find_token(plainSexp, "value", 0));
    if (!valueSexp)
        return std::nullopt;

    return mpiData(valueSexp);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmRSA_OAEP::platformEncrypt(const CryptoAlgorithmRsaOaepParams& parameters, const CryptoKeyRSA& key, const Vector<uint8_t>& plainText)
{
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(!(key.keySizeInBits() % 8));
    auto output = gcryptEncrypt(key.hashAlgorithmIdentifier(), key.platformKey(), parameters.labelVector(), plainText, key.keySizeInBits() / 8);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmRSA_OAEP::platformDecrypt(const CryptoAlgorithmRsaOaepParams& parameters, const CryptoKeyRSA& key, const Vector<uint8_t>& cipherText)
{
    auto output = gcryptDecrypt(key.hashAlgorithmIdentifier(), key.platformKey(), parameters.labelVector(), cipherText);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

} // namespace WebCore
