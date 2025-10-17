/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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
#include "CryptoAlgorithmECDH.h"

#include "CryptoKeyEC.h"
#include "GCryptUtilities.h"
#include <pal/crypto/gcrypt/Handle.h>
#include <pal/crypto/gcrypt/Utilities.h>

namespace WebCore {

static std::optional<Vector<uint8_t>> gcryptDerive(gcry_sexp_t baseKeySexp, gcry_sexp_t publicKeySexp, size_t keySizeInBytes)
{
    // First, retrieve private key data, which is roughly of the following form:
    // (private-key
    //   (ecc
    //     ...
    //     (d ...)))
    PAL::GCrypt::Handle<gcry_sexp_t> dataSexp;
    {
        PAL::GCrypt::Handle<gcry_sexp_t> dSexp(gcry_sexp_find_token(baseKeySexp, "d", 0));
        if (!dSexp)
            return std::nullopt;

        auto data = mpiData(dSexp);
        if (!data)
            return std::nullopt;

        gcry_sexp_build(&dataSexp, nullptr, "(data(flags raw)(value %b))", data->size(), data->data());
        if (!dataSexp)
            return std::nullopt;
    }

    // Encrypt the data s-expression with the public key.
    PAL::GCrypt::Handle<gcry_sexp_t> cipherSexp;
    gcry_error_t error = gcry_pk_encrypt(&cipherSexp, dataSexp, publicKeySexp);
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return std::nullopt;
    }

    // Retrieve the shared point value from the generated s-expression, which is of the following form:
    // (enc-val
    //   (ecdh
    //     (s ...)
    //     (e ...)))
    PAL::GCrypt::Handle<gcry_mpi_t> xMPI(gcry_mpi_new(0));
    if (!xMPI)
        return std::nullopt;

    {
        PAL::GCrypt::Handle<gcry_sexp_t> sSexp(gcry_sexp_find_token(cipherSexp, "s", 0));
        if (!sSexp)
            return std::nullopt;

        PAL::GCrypt::Handle<gcry_mpi_t> sMPI(gcry_sexp_nth_mpi(sSexp, 1, GCRYMPI_FMT_USG));
        if (!sMPI)
            return std::nullopt;

        PAL::GCrypt::Handle<gcry_mpi_point_t> point(gcry_mpi_point_new(0));
        if (!point)
            return std::nullopt;

        error = gcry_mpi_ec_decode_point(point, sMPI, nullptr);
        if (error != GPG_ERR_NO_ERROR)
            return std::nullopt;

        // We're only interested in the x-coordinate.
        gcry_mpi_point_snatch_get(xMPI, nullptr, nullptr, point.release());
    }

    return mpiZeroPrefixedData(xMPI, keySizeInBytes);
}

std::optional<Vector<uint8_t>> CryptoAlgorithmECDH::platformDeriveBits(const CryptoKeyEC& baseKey, const CryptoKeyEC& publicKey)
{
    return gcryptDerive(baseKey.platformKey().get(), publicKey.platformKey().get(), (baseKey.keySizeInBits() + 7) / 8);
}

} // namespace WebCore
