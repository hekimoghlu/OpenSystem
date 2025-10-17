/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
#include "GCryptRFC8032.h"

#include <pal/crypto/gcrypt/Handle.h>
#include <pal/crypto/gcrypt/Utilities.h>

namespace WebCore {
namespace GCrypt {
namespace RFC8032 {

bool validateEd25519KeyPair(const Vector<uint8_t>& privateKey, const Vector<uint8_t>& publicKey)
{
    if (privateKey.size() != 32 || publicKey.size() != 32)
        return false;

    gcry_error_t error = GPG_ERR_NO_ERROR;

    PAL::GCrypt::Handle<gcry_ctx_t> context;
    error = gcry_mpi_ec_new(&context, nullptr, "Ed25519");
    if (error != GPG_ERR_NO_ERROR) {
        PAL::GCrypt::logError(error);
        return false;
    }

    PAL::GCrypt::Handle<gcry_mpi_t> hdMPI;
    {
        // For Ed25519, the private-key data is hashed using SHA-512. The lower 32 bytes
        // are scanned as MPI data, meaning they also have to be reversed.
        std::array<uint8_t, 32> hddata;
        memcpy(hddata.data(), privateKey.data(), 32);

        std::array<uint8_t, 64> digest;
        gcry_md_hash_buffer(GCRY_MD_SHA512, digest.data(), hddata.data(), hddata.size());

        std::copy(std::next(digest.rbegin(), 32), digest.rend(), hddata.begin());
        error = gcry_mpi_scan(&hdMPI, GCRYMPI_FMT_USG, hddata.data(), hddata.size(), nullptr);
        if (error != GPG_ERR_NO_ERROR) {
            PAL::GCrypt::logError(error);
            return false;
        }

        // For Ed25519:
        //   - three least-significant bits of the first byte are set to zero,
        //   - second most-significant bit of the last byte is set to 1,
        //   - most-significant bit of the of the last byte is set to zero.
        gcry_mpi_clear_bit(hdMPI, 0);
        gcry_mpi_clear_bit(hdMPI, 1);
        gcry_mpi_clear_bit(hdMPI, 2);
        gcry_mpi_set_bit(hdMPI, 254);
        gcry_mpi_clear_bit(hdMPI, 255);
    }

    std::array<uint8_t, 32> qData;
    std::fill(qData.begin(), qData.end(), uint8_t(0));

    {
        // Perform the multiplication on the given curve. Both coordinates of the resulting point
        // are retrieved. The least-significant bit of the x-coordinate is copied into the
        // most-significant bit of the y-coordinate, which is then the public key value.
        PAL::GCrypt::Handle<gcry_mpi_point_t> G(gcry_mpi_ec_get_point("g", context, 1));
        PAL::GCrypt::Handle<gcry_mpi_point_t> Q(gcry_mpi_point_new(0));
        PAL::GCrypt::Handle<gcry_mpi_t> xMPI(gcry_mpi_new(0));
        PAL::GCrypt::Handle<gcry_mpi_t> yMPI(gcry_mpi_new(0));

        gcry_mpi_ec_mul(Q, hdMPI, G, context);
        int ret = gcry_mpi_ec_get_affine(xMPI, yMPI, Q, context);

        if (gcry_mpi_test_bit(xMPI, 0))
            gcry_mpi_set_bit(yMPI, 255);
        else
            gcry_mpi_clear_bit(yMPI, 255);

        // Store the resulting MPI data into a separate allocation. In case of an infinite point,
        // a zero Vector is established. Otherwise, the MPI data is retrieved into a properly-sized Vector.
        Vector<uint8_t> result;
        if (!ret) {
            size_t numBytes = 0;
            error = gcry_mpi_print(GCRYMPI_FMT_USG, nullptr, 0, &numBytes, yMPI);
            if (error != GPG_ERR_NO_ERROR) {
                PAL::GCrypt::logError(error);
                return false;
            }

            result = Vector<uint8_t>(numBytes, uint8_t(0));
            error = gcry_mpi_print(GCRYMPI_FMT_USG, result.data(), result.size(), nullptr, yMPI);
            if (error != GPG_ERR_NO_ERROR) {
                PAL::GCrypt::logError(error);
                return false;
            }
        } else
            result = Vector<uint8_t>(32, uint8_t(0));

        // Up to curve-specific amount of bytes of MPI data is copied in reverse order
        // into the initially-zeroed result Vector.
        size_t resultSize = std::min<size_t>(result.size(), 32);
        std::copy(result.rbegin(), std::next(result.rbegin(), resultSize), qData.begin());
    }

    return !memcmp(qData.begin(), publicKey.data(), 32);
}

} } } // namespace WebCore::GCrypt::RFC8032
