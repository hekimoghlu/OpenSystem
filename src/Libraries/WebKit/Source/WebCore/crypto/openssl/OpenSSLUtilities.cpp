/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
#include "OpenSSLUtilities.h"

#include "OpenSSLCryptoUniquePtr.h"

namespace WebCore {

const EVP_MD* digestAlgorithm(CryptoAlgorithmIdentifier hashFunction)
{
    switch (hashFunction) {
    case CryptoAlgorithmIdentifier::SHA_1:
        return EVP_sha1();
    case CryptoAlgorithmIdentifier::SHA_224:
        return EVP_sha224();
    case CryptoAlgorithmIdentifier::SHA_256:
        return EVP_sha256();
    case CryptoAlgorithmIdentifier::SHA_384:
        return EVP_sha384();
    case CryptoAlgorithmIdentifier::SHA_512:
        return EVP_sha512();
    default:
        return nullptr;
    }
}

std::optional<Vector<uint8_t>> calculateDigest(const EVP_MD* algorithm, const Vector<uint8_t>& message)
{
    EvpDigestCtxPtr ctx;
    if (!(ctx = EvpDigestCtxPtr(EVP_MD_CTX_create())))
        return std::nullopt;

    int digestLength = EVP_MD_size(algorithm);
    if (digestLength <= 0)
        return std::nullopt;
    Vector<uint8_t> digest(digestLength);

    if (EVP_DigestInit_ex(ctx.get(), algorithm, nullptr) != 1)
        return std::nullopt;

    if (EVP_DigestUpdate(ctx.get(), message.data(), message.size()) != 1)
        return std::nullopt;

    if (EVP_DigestFinal_ex(ctx.get(), digest.data(), nullptr) != 1)
        return std::nullopt;

    return digest;
}

Vector<uint8_t> convertToBytes(const BIGNUM* bignum)
{
    Vector<uint8_t> bytes(BN_num_bytes(bignum));
    BN_bn2bin(bignum, bytes.data());
    return bytes;
}

Vector<uint8_t> convertToBytesExpand(const BIGNUM* bignum, size_t minimumBufferSize)
{
    int length = BN_num_bytes(bignum);
    if (length < 0)
        return { };

    size_t bufferSize = std::max<size_t>(length, minimumBufferSize);

    Vector<uint8_t> bytes(bufferSize);

    size_t paddingLength = bufferSize - length;
    if (paddingLength > 0) {
        uint8_t padding = BN_is_negative(bignum) ? 0xFF : 0x00;
        for (size_t i = 0; i < paddingLength; i++)
            bytes[i] = padding;
    }
    BN_bn2bin(bignum, bytes.data() + paddingLength);
    return bytes;
}

BIGNUMPtr convertToBigNumber(const Vector<uint8_t>& bytes)
{
    return BIGNUMPtr(BN_bin2bn(bytes.data(), bytes.size(), nullptr));
}

bool AESKey::setKey(const Vector<uint8_t>& key, int enc)
{
    size_t keySize = key.size() * 8;
    if (keySize != 128 && keySize != 192 && keySize != 256)
        return false;

    if (enc == AES_ENCRYPT) {
        if (AES_set_encrypt_key(key.data(), keySize, &m_key) < 0)
            return false;
        return true;
    }

    if (enc == AES_DECRYPT) {
        if (AES_set_decrypt_key(key.data(), keySize, &m_key) < 0)
            return false;
        return true;
    }

    ASSERT_NOT_REACHED();
    return false;
}

AESKey::~AESKey()
{
    memset(&m_key, 0, sizeof m_key);
}

} // namespace WebCore
