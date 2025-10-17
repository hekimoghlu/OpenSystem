/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#include "GCryptUtilities.h"


namespace WebCore {

std::optional<ASCIILiteral> hashAlgorithmName(CryptoAlgorithmIdentifier identifier)
{
    switch (identifier) {
    case CryptoAlgorithmIdentifier::SHA_1:
        return "sha1"_s;
    case CryptoAlgorithmIdentifier::SHA_224:
        return "sha224"_s;
    case CryptoAlgorithmIdentifier::SHA_256:
        return "sha256"_s;
    case CryptoAlgorithmIdentifier::SHA_384:
        return "sha384"_s;
    case CryptoAlgorithmIdentifier::SHA_512:
        return "sha512"_s;
    default:
        return std::nullopt;
    }
}

std::optional<int> hmacAlgorithm(CryptoAlgorithmIdentifier identifier)
{
    switch (identifier) {
    case CryptoAlgorithmIdentifier::SHA_1:
        return GCRY_MAC_HMAC_SHA1;
    case CryptoAlgorithmIdentifier::SHA_224:
        return GCRY_MAC_HMAC_SHA224;
    case CryptoAlgorithmIdentifier::SHA_256:
        return GCRY_MAC_HMAC_SHA256;
    case CryptoAlgorithmIdentifier::SHA_384:
        return GCRY_MAC_HMAC_SHA384;
    case CryptoAlgorithmIdentifier::SHA_512:
        return GCRY_MAC_HMAC_SHA512;
    default:
        return std::nullopt;
    }
}

std::optional<int> digestAlgorithm(CryptoAlgorithmIdentifier identifier)
{
    switch (identifier) {
    case CryptoAlgorithmIdentifier::SHA_1:
        return GCRY_MD_SHA1;
    case CryptoAlgorithmIdentifier::SHA_224:
        return GCRY_MD_SHA224;
    case CryptoAlgorithmIdentifier::SHA_256:
        return GCRY_MD_SHA256;
    case CryptoAlgorithmIdentifier::SHA_384:
        return GCRY_MD_SHA384;
    case CryptoAlgorithmIdentifier::SHA_512:
        return GCRY_MD_SHA512;
    default:
        return std::nullopt;
    }
}

std::optional<PAL::CryptoDigest::Algorithm> hashCryptoDigestAlgorithm(CryptoAlgorithmIdentifier identifier)
{
    switch (identifier) {
    case CryptoAlgorithmIdentifier::SHA_1:
        return PAL::CryptoDigest::Algorithm::SHA_1;
    case CryptoAlgorithmIdentifier::SHA_224:
        return PAL::CryptoDigest::Algorithm::SHA_224;
    case CryptoAlgorithmIdentifier::SHA_256:
        return PAL::CryptoDigest::Algorithm::SHA_256;
    case CryptoAlgorithmIdentifier::SHA_384:
        return PAL::CryptoDigest::Algorithm::SHA_384;
    case CryptoAlgorithmIdentifier::SHA_512:
        return PAL::CryptoDigest::Algorithm::SHA_512;
    default:
        return std::nullopt;
    }
}

std::optional<size_t> mpiLength(gcry_mpi_t paramMPI)
{
    // Retrieve the MPI length for the unsigned format.
    size_t dataLength = 0;
    gcry_error_t error = gcry_mpi_print(GCRYMPI_FMT_USG, nullptr, 0, &dataLength, paramMPI);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    return dataLength;
}

std::optional<size_t> mpiLength(gcry_sexp_t paramSexp)
{
    // Retrieve the MPI value stored in the s-expression: (name mpi-data)
    PAL::GCrypt::Handle<gcry_mpi_t> paramMPI(gcry_sexp_nth_mpi(paramSexp, 1, GCRYMPI_FMT_USG));
    if (!paramMPI)
        return std::nullopt;

    return mpiLength(paramMPI);
}

std::optional<Vector<uint8_t>> mpiData(gcry_mpi_t paramMPI)
{
    // Retrieve the MPI length.
    auto length = mpiLength(paramMPI);
    if (!length)
        return std::nullopt;

    // Copy the MPI data into a properly-sized buffer.
    Vector<uint8_t> output(*length);
    gcry_error_t error = gcry_mpi_print(GCRYMPI_FMT_USG, output.data(), output.size(), nullptr, paramMPI);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    return output;
}

std::optional<Vector<uint8_t>> mpiZeroPrefixedData(gcry_mpi_t paramMPI, size_t targetLength)
{
    // Retrieve the MPI length. Bail if the retrieved length is longer than target length.
    auto length = mpiLength(paramMPI);
    if (!length || *length > targetLength)
        return std::nullopt;

    // Fill out the output buffer with zeros. Properly determine the zero prefix length,
    // and copy the MPI data into memory area following the prefix (if any).
    Vector<uint8_t> output(targetLength, 0);
    size_t prefixLength = targetLength - *length;
    gcry_error_t error = gcry_mpi_print(GCRYMPI_FMT_USG, const_cast<uint8_t*>(output.subspan(prefixLength).data()), targetLength, nullptr, paramMPI);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    return output;
}

std::optional<Vector<uint8_t>> mpiData(gcry_sexp_t paramSexp)
{
    // Retrieve the MPI value stored in the s-expression: (name mpi-data)
    PAL::GCrypt::Handle<gcry_mpi_t> paramMPI(gcry_sexp_nth_mpi(paramSexp, 1, GCRYMPI_FMT_USG));
    if (!paramMPI)
        return std::nullopt;

    return mpiData(paramMPI);
}

std::optional<Vector<uint8_t>> mpiZeroPrefixedData(gcry_sexp_t paramSexp, size_t targetLength)
{
    // Retrieve the MPI value stored in the s-expression: (name mpi-data)
    PAL::GCrypt::Handle<gcry_mpi_t> paramMPI(gcry_sexp_nth_mpi(paramSexp, 1, GCRYMPI_FMT_USG));
    if (!paramMPI)
        return std::nullopt;

    return mpiZeroPrefixedData(paramMPI, targetLength);
}

std::optional<Vector<uint8_t>> mpiSignedData(gcry_mpi_t mpi)
{
    auto data = mpiData(mpi);
    if (!data)
        return std::nullopt;

    if (data->at(0) & 0x80)
        data->insert(0, 0x00);

    return data;
}

std::optional<Vector<uint8_t>> mpiSignedData(gcry_sexp_t paramSexp)
{
    auto data = mpiData(paramSexp);
    if (!data)
        return std::nullopt;

    if (data->at(0) & 0x80)
        data->insert(0, 0x00);

    return data;
}

} // namespace WebCore
