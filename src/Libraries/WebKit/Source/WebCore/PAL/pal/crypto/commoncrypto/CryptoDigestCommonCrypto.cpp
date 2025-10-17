/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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

#if HAVE(SWIFT_CPP_INTEROP)
#include "PALSwift.h"
#endif
#include <CommonCrypto/CommonCrypto.h>
#include <optional>
#include <wtf/TZoneMallocInlines.h>

namespace PAL {

struct CryptoDigestContext {
    WTF_MAKE_STRUCT_TZONE_ALLOCATED(CryptoDigestContext);
    CryptoDigest::Algorithm algorithm;
    std::variant<
#if HAVE(SWIFT_CPP_INTEROP)
        std::unique_ptr<PAL::Digest>,
#endif
        std::unique_ptr<CC_SHA1_CTX>,
        std::unique_ptr<CC_SHA256_CTX>,
        std::unique_ptr<CC_SHA512_CTX>
    > ccContext;
};

WTF_MAKE_STRUCT_TZONE_ALLOCATED_IMPL(CryptoDigestContext);

inline CC_SHA1_CTX* toSHA1Context(CryptoDigestContext* context)
{
    ASSERT(context->algorithm == CryptoDigest::Algorithm::SHA_1);
    return static_cast<CC_SHA1_CTX*>(std::get<std::unique_ptr<CC_SHA1_CTX>>(context->ccContext).get());
}
inline CC_SHA256_CTX* toSHA224Context(CryptoDigestContext* context)
{
    ASSERT(context->algorithm == CryptoDigest::Algorithm::SHA_224);
    return static_cast<CC_SHA256_CTX*>(std::get<std::unique_ptr<CC_SHA256_CTX>>(context->ccContext).get());
}
inline CC_SHA256_CTX* toSHA256Context(CryptoDigestContext* context)
{
    ASSERT(context->algorithm == CryptoDigest::Algorithm::SHA_256);
    return static_cast<CC_SHA256_CTX*>(std::get<std::unique_ptr<CC_SHA256_CTX>>(context->ccContext).get());
}
inline CC_SHA512_CTX* toSHA384Context(CryptoDigestContext* context)
{
    ASSERT(context->algorithm == CryptoDigest::Algorithm::SHA_384);
    return static_cast<CC_SHA512_CTX*>(std::get<std::unique_ptr<CC_SHA512_CTX>>(context->ccContext).get());
}
inline CC_SHA512_CTX* toSHA512Context(CryptoDigestContext* context)
{
    ASSERT(context->algorithm == CryptoDigest::Algorithm::SHA_512);
    return static_cast<CC_SHA512_CTX*>(std::get<std::unique_ptr<CC_SHA512_CTX>>(context->ccContext).get());
}

CryptoDigest::CryptoDigest()
    : m_context(WTF::makeUnique<CryptoDigestContext>())
{
}

CryptoDigest::~CryptoDigest()
{
}

#if HAVE(SWIFT_CPP_INTEROP)
static std::variant<std::unique_ptr<PAL::Digest>, std::unique_ptr<CC_SHA1_CTX>, std::unique_ptr<CC_SHA256_CTX>, std::unique_ptr<CC_SHA512_CTX>> createCryptoDigest(CryptoDigest::Algorithm algorithm)
#else
static std::variant<std::unique_ptr<CC_SHA1_CTX>, std::unique_ptr<CC_SHA256_CTX>, std::unique_ptr<CC_SHA512_CTX>> createCryptoDigest(CryptoDigest::Algorithm algorithm)
#endif
{
    switch (algorithm) {
#if !HAVE(SWIFT_CPP_INTEROP)
    case CryptoDigest::Algorithm::SHA_1: {
        std::unique_ptr<CC_SHA1_CTX> context = WTF::makeUniqueWithoutFastMallocCheck<CC_SHA1_CTX>();
        ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        CC_SHA1_Init(context.get());
        ALLOW_DEPRECATED_DECLARATIONS_END
        return context;
    }
    case CryptoDigest::Algorithm::SHA_256: {
        std::unique_ptr<CC_SHA256_CTX> context = WTF::makeUniqueWithoutFastMallocCheck<CC_SHA256_CTX>();
        CC_SHA256_Init(context.get());
        return context;
    }
    case CryptoDigest::Algorithm::SHA_384: {
        std::unique_ptr<CC_SHA512_CTX> context = WTF::makeUniqueWithoutFastMallocCheck<CC_SHA512_CTX>();
        CC_SHA384_Init(context.get());
        return context;
    }
    case CryptoDigest::Algorithm::SHA_512: {
        std::unique_ptr<CC_SHA512_CTX> context = WTF::makeUniqueWithoutFastMallocCheck<CC_SHA512_CTX>();
        CC_SHA512_Init(context.get());
        return context;
    }
#else
    case CryptoDigest::Algorithm::SHA_1: {
        return WTF::makeUniqueWithoutFastMallocCheck<PAL::Digest>(PAL::Digest::sha1Init());
    }
    case CryptoDigest::Algorithm::SHA_256: {
        return WTF::makeUniqueWithoutFastMallocCheck<PAL::Digest>(PAL::Digest::sha256Init());
    }
    case CryptoDigest::Algorithm::SHA_384: {
        return WTF::makeUniqueWithoutFastMallocCheck<PAL::Digest>(PAL::Digest::sha384Init());
    }
    case CryptoDigest::Algorithm::SHA_512: {
        return WTF::makeUniqueWithoutFastMallocCheck<PAL::Digest>(PAL::Digest::sha512Init());
    }
#endif
    case CryptoDigest::Algorithm::SHA_224: {
        std::unique_ptr<CC_SHA256_CTX> context = WTF::makeUniqueWithoutFastMallocCheck<CC_SHA256_CTX>();
        CC_SHA224_Init(context.get());
        return context;
    }
    }
}

std::unique_ptr<CryptoDigest> CryptoDigest::create(CryptoDigest::Algorithm algorithm)
{
    std::unique_ptr<CryptoDigest> digest = WTF::makeUnique<CryptoDigest>();
    ASSERT(digest->m_context);
    digest->m_context->algorithm = algorithm;
    digest->m_context->ccContext = createCryptoDigest(algorithm);
    return digest;
}

void CryptoDigest::addBytes(std::span<const uint8_t> input)
{
    switch (m_context->algorithm) {
#if !HAVE(SWIFT_CPP_INTEROP)
    case CryptoDigest::Algorithm::SHA_1:
        ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        CC_SHA1_Update(toSHA1Context(m_context.get()), static_cast<const void*>(input.data()), input.size());
        ALLOW_DEPRECATED_DECLARATIONS_END
        return;
    case CryptoDigest::Algorithm::SHA_256:
        CC_SHA256_Update(toSHA256Context(m_context.get()), static_cast<const void*>(input.data()), input.size());
        return;
    case CryptoDigest::Algorithm::SHA_384:
        CC_SHA384_Update(toSHA384Context(m_context.get()), static_cast<const void*>(input.data()), input.size());
        return;
    case CryptoDigest::Algorithm::SHA_512:
        CC_SHA512_Update(toSHA512Context(m_context.get()), static_cast<const void*>(input.data()), input.size());
        return;
#else
    case CryptoDigest::Algorithm::SHA_1:
    case CryptoDigest::Algorithm::SHA_256:
    case CryptoDigest::Algorithm::SHA_384:
    case CryptoDigest::Algorithm::SHA_512:
        std::get<std::unique_ptr<PAL::Digest>>(m_context->ccContext)->update(input);
        return;
#endif
    case CryptoDigest::Algorithm::SHA_224:
        CC_SHA224_Update(toSHA224Context(m_context.get()), static_cast<const void*>(input.data()), input.size());
        return;
    }
}

Vector<uint8_t> CryptoDigest::computeHash()
{
    Vector<uint8_t> result;
    switch (m_context->algorithm) {
#if !HAVE(SWIFT_CPP_INTEROP)
    case CryptoDigest::Algorithm::SHA_1:
        result.grow(CC_SHA1_DIGEST_LENGTH);
        ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        CC_SHA1_Final(result.data(), toSHA1Context(m_context.get()));
        ALLOW_DEPRECATED_DECLARATIONS_END
        break;
    case CryptoDigest::Algorithm::SHA_256:
        result.grow(CC_SHA256_DIGEST_LENGTH);
        CC_SHA256_Final(result.data(), toSHA256Context(m_context.get()));
        break;
    case CryptoDigest::Algorithm::SHA_384:
        result.grow(CC_SHA384_DIGEST_LENGTH);
        CC_SHA384_Final(result.data(), toSHA384Context(m_context.get()));
        break;
    case CryptoDigest::Algorithm::SHA_512:
        result.grow(CC_SHA512_DIGEST_LENGTH);
        CC_SHA512_Final(result.data(), toSHA512Context(m_context.get()));
        break;
#else
    case CryptoDigest::Algorithm::SHA_1:
    case CryptoDigest::Algorithm::SHA_256:
    case CryptoDigest::Algorithm::SHA_384:
    case CryptoDigest::Algorithm::SHA_512:
        return std::get<std::unique_ptr<PAL::Digest>>(m_context->ccContext)->finalize();
#endif
    case CryptoDigest::Algorithm::SHA_224:
        result.grow(CC_SHA224_DIGEST_LENGTH);
        CC_SHA224_Final(result.data(), toSHA224Context(m_context.get()));
        break;

    }
    return result;
}

} // namespace PAL
