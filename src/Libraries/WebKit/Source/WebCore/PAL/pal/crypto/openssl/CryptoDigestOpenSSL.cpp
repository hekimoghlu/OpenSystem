/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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

#include <openssl/sha.h>

namespace {
struct SHA1Functions {
    static constexpr auto init = SHA1_Init;
    static constexpr auto update = SHA1_Update;
    static constexpr auto final = SHA1_Final;
    static constexpr size_t digestLength = SHA_DIGEST_LENGTH;
};

struct SHA224Functions {
    static constexpr auto init = SHA224_Init;
    static constexpr auto update = SHA224_Update;
    static constexpr auto final = SHA224_Final;
    static constexpr size_t digestLength = SHA224_DIGEST_LENGTH;
};

struct SHA256Functions {
    static constexpr auto init = SHA256_Init;
    static constexpr auto update = SHA256_Update;
    static constexpr auto final = SHA256_Final;
    static constexpr size_t digestLength = SHA256_DIGEST_LENGTH;
};

struct SHA384Functions {
    static constexpr auto init = SHA384_Init;
    static constexpr auto update = SHA384_Update;
    static constexpr auto final = SHA384_Final;
    static constexpr size_t digestLength = SHA384_DIGEST_LENGTH;
};

struct SHA512Functions {
    static constexpr auto init = SHA512_Init;
    static constexpr auto update = SHA512_Update;
    static constexpr auto final = SHA512_Final;
    static constexpr size_t digestLength = SHA512_DIGEST_LENGTH;
};
}

namespace PAL {

struct CryptoDigestContext {
    virtual ~CryptoDigestContext() = default;
    virtual void addBytes(std::span<const uint8_t> input) = 0;
    virtual Vector<uint8_t> computeHash() = 0;
};

template <typename SHAContext, typename SHAFunctions>
struct CryptoDigestContextImpl : public CryptoDigestContext {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    static std::unique_ptr<CryptoDigestContext> create()
    {
        return makeUnique<CryptoDigestContextImpl>();
    }

    CryptoDigestContextImpl()
    {
        SHAFunctions::init(&m_context);
    }

    void addBytes(std::span<const uint8_t> input) override
    {
        SHAFunctions::update(&m_context, static_cast<const void*>(input.data()), input.size());
    }

    Vector<uint8_t> computeHash() override
    {
        Vector<uint8_t> result(SHAFunctions::digestLength);
        SHAFunctions::final(result.data(), &m_context);
        return result;
    }

private:
    SHAContext m_context;
};

CryptoDigest::CryptoDigest() = default;

CryptoDigest::~CryptoDigest() = default;

static std::unique_ptr<CryptoDigestContext> createCryptoDigest(CryptoDigest::Algorithm algorithm)
{
    switch (algorithm) {
    case CryptoDigest::Algorithm::SHA_1:
        return CryptoDigestContextImpl<SHA_CTX, SHA1Functions>::create();
    case CryptoDigest::Algorithm::SHA_224:
        return CryptoDigestContextImpl<SHA256_CTX, SHA224Functions>::create();
    case CryptoDigest::Algorithm::SHA_256:
        return CryptoDigestContextImpl<SHA256_CTX, SHA256Functions>::create();
    case CryptoDigest::Algorithm::SHA_384:
        return CryptoDigestContextImpl<SHA512_CTX, SHA384Functions>::create();
    case CryptoDigest::Algorithm::SHA_512:
        return CryptoDigestContextImpl<SHA512_CTX, SHA512Functions>::create();
    }
    return nullptr;
}

std::unique_ptr<CryptoDigest> CryptoDigest::create(CryptoDigest::Algorithm algorithm)
{
    std::unique_ptr<CryptoDigest> digest = WTF::makeUnique<CryptoDigest>();
    digest->m_context = createCryptoDigest(algorithm);
    return digest;
}

void CryptoDigest::addBytes(std::span<const uint8_t> input)
{
    ASSERT(m_context);
    m_context->addBytes(input);
}

Vector<uint8_t> CryptoDigest::computeHash()
{
    ASSERT(m_context);
    return m_context->computeHash();
}

} // namespace PAL
