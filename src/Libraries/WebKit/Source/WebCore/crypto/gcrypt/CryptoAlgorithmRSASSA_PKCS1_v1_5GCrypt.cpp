/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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
#include "CryptoAlgorithmRSASSA_PKCS1_v1_5.h"

#include "CryptoKeyRSA.h"
#include "GCryptUtilities.h"
#include "NotImplemented.h"

namespace WebCore {

static std::optional<Vector<uint8_t>> gcryptSign(gcry_sexp_t keySexp, const Vector<uint8_t>& data, CryptoAlgorithmIdentifier hashAlgorithmIdentifier, size_t keySizeInBytes)
{
    // Perform digest operation with the specified algorithm on the given data.
    Vector<uint8_t> dataHash;
    {
        auto digestAlgorithm = hashCryptoDigestAlgorithm(hashAlgorithmIdentifier);
        if (!digestAlgorithm)
            return std::nullopt;

        auto digest = PAL::CryptoDigest::create(*digestAlgorithm);
        if (!digest)
            return std::nullopt;

        digest->addBytes(data);
        dataHash = digest->computeHash();
    }

    // Construct the data s-expression that contains PKCS#1-padded hashed data.
    PAL::GCrypt::Handle<gcry_sexp_t> dataSexp;
    {
        auto shaAlgorithm = hashAlgorithmName(hashAlgorithmIdentifier);
        if (!shaAlgorithm)
            return std::nullopt;

        gcry_error_t error = gcry_sexp_build(&dataSexp, nullptr, "(data(flags pkcs1)(hash %s %b))",
            shaAlgorithm->characters(), dataHash.size(), dataHash.data());
        if (error != GPG_ERR_NO_ERROR) {
            PAL::GCrypt::logError(error);
            return std::nullopt;
        }
    }

    // Perform the PK signing, retrieving a sig-val s-expression of the following form:
    // (sig-val
    //   (rsa
    //     (s s-mpi)))
    PAL::GCrypt::Handle<gcry_sexp_t> signatureSexp;
    gcry_error_t error = gcry_pk_sign(&signatureSexp, dataSexp, keySexp);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Return MPI data of the embedded s integer.
    PAL::GCrypt::Handle<gcry_sexp_t> sSexp(gcry_sexp_find_token(signatureSexp, "s", 0));
    if (!sSexp)
        return std::nullopt;

    return mpiZeroPrefixedData(sSexp, keySizeInBytes);
}

static std::optional<bool> gcryptVerify(gcry_sexp_t keySexp, const Vector<uint8_t>& signature, const Vector<uint8_t>& data, CryptoAlgorithmIdentifier hashAlgorithmIdentifier)
{
    // Perform digest operation with the specified algorithm on the given data.
    Vector<uint8_t> dataHash;
    {
        auto digestAlgorithm = hashCryptoDigestAlgorithm(hashAlgorithmIdentifier);
        if (!digestAlgorithm)
            return std::nullopt;

        auto digest = PAL::CryptoDigest::create(*digestAlgorithm);
        if (!digest)
            return std::nullopt;

        digest->addBytes(data);
        dataHash = digest->computeHash();
    }

    // Construct the sig-val s-expression that contains the signature data.
    PAL::GCrypt::Handle<gcry_sexp_t> signatureSexp;
    gcry_error_t error = gcry_sexp_build(&signatureSexp, nullptr, "(sig-val(rsa(s %b)))",
        signature.size(), signature.data());
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Construct the data s-expression that contains PKCS#1-padded hashed data.
    PAL::GCrypt::Handle<gcry_sexp_t> dataSexp;
    {
        auto shaAlgorithm = hashAlgorithmName(hashAlgorithmIdentifier);
        if (!shaAlgorithm)
            return std::nullopt;

        error = gcry_sexp_build(&dataSexp, nullptr, "(data(flags pkcs1)(hash %s %b))",
            shaAlgorithm->characters(), dataHash.size(), dataHash.data());
        if (error != GPG_ERR_NO_ERROR) {
            PAL::GCrypt::logError(error);
            return std::nullopt;
        }
    }

    // Perform the PK verification. We report success if there's no error returned, or
    // a failure in any other case. OperationError should not be returned at this point,
    // avoiding spilling information about the exact cause of verification failure.
    error = gcry_pk_verify(signatureSexp, dataSexp, keySexp);
    return { error == GPG_ERR_NO_ERROR };
}

ExceptionOr<Vector<uint8_t>> CryptoAlgorithmRSASSA_PKCS1_v1_5::platformSign(const CryptoKeyRSA& key, const Vector<uint8_t>& data)
{
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(!(key.keySizeInBits() % 8));
    auto output = gcryptSign(key.platformKey(), data, key.hashAlgorithmIdentifier(), key.keySizeInBits() / 8);
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return WTFMove(*output);
}

ExceptionOr<bool> CryptoAlgorithmRSASSA_PKCS1_v1_5::platformVerify(const CryptoKeyRSA& key, const Vector<uint8_t>& signature, const Vector<uint8_t>& data)
{
    auto output = gcryptVerify(key.platformKey(), signature, data, key.hashAlgorithmIdentifier());
    if (!output)
        return Exception { ExceptionCode::OperationError };
    return *output;
}

} // namespace WebCore
