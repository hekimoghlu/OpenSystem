/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include "CryptoDigest.h"

#include <gcrypt.h>

namespace PAL {

struct CryptoDigestContext {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    int algorithm;
    gcry_md_hd_t md;
};

CryptoDigest::CryptoDigest()
    : m_context(WTF::makeUnique<CryptoDigestContext>())
{
}

CryptoDigest::~CryptoDigest() = default;

static int getGcryptAlgorithm(CryptoDigest::Algorithm algorithm)
{
    switch (algorithm) {
    case CryptoDigest::Algorithm::SHA_1:
        return GCRY_MD_SHA1;
    case CryptoDigest::Algorithm::SHA_224:
        return GCRY_MD_SHA224;
    case CryptoDigest::Algorithm::SHA_256:
        return GCRY_MD_SHA256;
    case CryptoDigest::Algorithm::SHA_384:
        return GCRY_MD_SHA384;
    case CryptoDigest::Algorithm::SHA_512:
        return GCRY_MD_SHA512;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return GCRY_MD_SHA256;

}

std::unique_ptr<CryptoDigest> CryptoDigest::create(CryptoDigest::Algorithm algorithm)
{
    int gcryptAlgorithm = getGcryptAlgorithm(algorithm);

    std::unique_ptr<CryptoDigest> digest = WTF::makeUnique<CryptoDigest>();
    digest->m_context->algorithm = gcryptAlgorithm;

    gcry_md_open(&digest->m_context->md, gcryptAlgorithm, 0);
    if (!digest->m_context->md)
        return nullptr;

    return digest;
}

void CryptoDigest::addBytes(std::span<const uint8_t> input)
{
    gcry_md_write(m_context->md, static_cast<const void*>(input.data()), input.size());
}

Vector<uint8_t> CryptoDigest::computeHash()
{
    size_t digestLen = gcry_md_get_algo_dlen(m_context->algorithm);

    gcry_md_final(m_context->md);
    Vector<uint8_t> result(unsafeMakeSpan<uint8_t>(gcry_md_read(m_context->md, 0), digestLen));
    gcry_md_close(m_context->md);

    return result;
}

} // namespace PAL
