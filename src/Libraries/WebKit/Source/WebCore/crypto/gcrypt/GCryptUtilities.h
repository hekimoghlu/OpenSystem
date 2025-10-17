/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 18, 2023.
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
#pragma once

#include "CryptoAlgorithmIdentifier.h"
#include <array>
#include <cstring>
#include <gcrypt.h>
#include <pal/crypto/CryptoDigest.h>
#include <pal/crypto/gcrypt/Handle.h>
#include <pal/crypto/gcrypt/Utilities.h>

namespace WebCore {

namespace CryptoConstants {

static const std::array<uint8_t, 12> s_ed25519Identifier { { "1.3.101.112" } };
static const std::array<uint8_t, 12> s_x25519Identifier { { "1.3.101.110" } };

static const std::array<uint8_t, 18> s_ecPublicKeyIdentifier { { "1.2.840.10045.2.1" } };
static const std::array<uint8_t, 13> s_ecDHIdentifier { { "1.3.132.1.12" } };

static const std::array<uint8_t, 20> s_secp256r1Identifier { { "1.2.840.10045.3.1.7" } };
static const std::array<uint8_t, 13> s_secp384r1Identifier { { "1.3.132.0.34" } };
static const std::array<uint8_t, 13> s_secp521r1Identifier { { "1.3.132.0.35" } };

static const std::array<uint8_t, 21> s_rsaEncryptionIdentifier { { "1.2.840.113549.1.1.1" } };
static const std::array<uint8_t, 21> s_RSAES_OAEPIdentifier { { "1.2.840.113549.1.1.7" } };
static const std::array<uint8_t, 22> s_RSASSA_PSSIdentifier { { "1.2.840.113549.1.1.10" } };

static const std::array<uint8_t, 2> s_asn1NullValue { { 0x05, 0x00 } };
static const std::array<uint8_t, 1> s_asn1Version0 { { 0x00 } };
static const std::array<uint8_t, 1> s_asn1Version1 { { 0x01 } };

static const std::array<uint8_t, 1> s_ecUncompressedFormatLeadingByte { { 0x04 } };
static const std::array<uint8_t, 1> s_x25519UncompressedFormatLeadingByte { { 0x40 } };

template<size_t N>
static inline bool matches(const void* lhs, size_t size, const std::array<uint8_t, N>& rhs)
{
    if (size != rhs.size())
        return false;

    return !std::memcmp(lhs, rhs.data(), rhs.size());
}

} // namespace CryptoConstants

std::optional<ASCIILiteral> hashAlgorithmName(CryptoAlgorithmIdentifier);

std::optional<int> hmacAlgorithm(CryptoAlgorithmIdentifier);
std::optional<int> digestAlgorithm(CryptoAlgorithmIdentifier);
std::optional<PAL::CryptoDigest::Algorithm> hashCryptoDigestAlgorithm(CryptoAlgorithmIdentifier);

std::optional<size_t> mpiLength(gcry_mpi_t);
std::optional<size_t> mpiLength(gcry_sexp_t);
std::optional<Vector<uint8_t>> mpiData(gcry_mpi_t);
std::optional<Vector<uint8_t>> mpiZeroPrefixedData(gcry_mpi_t, size_t targetLength);
std::optional<Vector<uint8_t>> mpiData(gcry_sexp_t);
std::optional<Vector<uint8_t>> mpiZeroPrefixedData(gcry_sexp_t, size_t targetLength);
std::optional<Vector<uint8_t>> mpiSignedData(gcry_mpi_t);
std::optional<Vector<uint8_t>> mpiSignedData(gcry_sexp_t);

} // namespace WebCore
