/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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

#include <windows.h>
#include <wincrypt.h>

namespace PAL {

struct CryptoDigestContext {
    CryptoDigest::Algorithm algorithm;
    HCRYPTPROV hContext { 0 };
    HCRYPTHASH hHash { 0 };
};

CryptoDigest::CryptoDigest()
    : m_context(new CryptoDigestContext)
{
}

CryptoDigest::~CryptoDigest()
{
    if (HCRYPTHASH hHash = m_context->hHash)
        CryptDestroyHash(hHash);
    if (HCRYPTPROV hContext = m_context->hContext)
        CryptReleaseContext(hContext, 0);
}

std::unique_ptr<CryptoDigest> CryptoDigest::create(Algorithm algorithm)
{
    std::unique_ptr<CryptoDigest> digest(new CryptoDigest);
    digest->m_context->algorithm = algorithm;
    if (!CryptAcquireContext(&digest->m_context->hContext, nullptr, nullptr /* use default provider */, PROV_RSA_AES, CRYPT_VERIFYCONTEXT))
        return nullptr;
    bool succeeded = false;
    switch (algorithm) {
    case CryptoDigest::Algorithm::SHA_1: {
        succeeded = CryptCreateHash(digest->m_context->hContext, CALG_SHA1, 0, 0, &digest->m_context->hHash);
        break;
    }
    case CryptoDigest::Algorithm::SHA_256: {
        succeeded = CryptCreateHash(digest->m_context->hContext, CALG_SHA_256, 0, 0, &digest->m_context->hHash);
        break;
    }
    case CryptoDigest::Algorithm::SHA_384: {
        succeeded = CryptCreateHash(digest->m_context->hContext, CALG_SHA_384, 0, 0, &digest->m_context->hHash);
        break;
    }
    case CryptoDigest::Algorithm::SHA_512: {
        succeeded = CryptCreateHash(digest->m_context->hContext, CALG_SHA_512, 0, 0, &digest->m_context->hHash);
        break;
    }
    }
    if (succeeded)
        return digest;
    return nullptr;
}

void CryptoDigest::addBytes(std::span<const uint8_t> input)
{
    if (input.empty())
        return;
    RELEASE_ASSERT(CryptHashData(m_context->hHash, reinterpret_cast<const BYTE*>(input.data()), input.size(), 0));
}

Vector<uint8_t> CryptoDigest::computeHash()
{
    Vector<uint8_t> result;
    DWORD digestLengthBuffer;
    DWORD digestLengthBufferSize = sizeof(digestLengthBuffer);

    RELEASE_ASSERT(CryptGetHashParam(m_context->hHash, HP_HASHSIZE, reinterpret_cast<BYTE*>(&digestLengthBuffer), &digestLengthBufferSize, 0));
    result.resize(digestLengthBuffer);

    RELEASE_ASSERT(CryptGetHashParam(m_context->hHash, HP_HASHVAL, result.data(), &digestLengthBuffer, 0));
    RELEASE_ASSERT(result.size() == digestLengthBuffer);
    return result;
}

} // namespace PAL
