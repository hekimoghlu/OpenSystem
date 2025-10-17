/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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
#include "ResourceCryptographicDigest.h"

#include "SharedBuffer.h"
#include <pal/crypto/CryptoDigest.h>
#include <wtf/text/Base64.h>
#include <wtf/text/ParsingUtilities.h>
#include <wtf/text/StringParsingBuffer.h>

namespace WebCore {

template<typename CharacterType> static std::optional<ResourceCryptographicDigest::Algorithm> parseHashAlgorithmAdvancingPosition(StringParsingBuffer<CharacterType>& buffer)
{
    // FIXME: This would be much cleaner with a lookup table of pairs of label / algorithm enum values, but I can't
    // figure out how to keep the labels compiletime strings for skipExactlyIgnoringASCIICase.

    if (skipExactlyIgnoringASCIICase(buffer, "sha256"_s))
        return ResourceCryptographicDigest::Algorithm::SHA256;
    if (skipExactlyIgnoringASCIICase(buffer, "sha384"_s))
        return ResourceCryptographicDigest::Algorithm::SHA384;
    if (skipExactlyIgnoringASCIICase(buffer, "sha512"_s))
        return ResourceCryptographicDigest::Algorithm::SHA512;

    return std::nullopt;
}

template<typename CharacterType> static std::optional<ResourceCryptographicDigest> parseCryptographicDigestImpl(StringParsingBuffer<CharacterType>& buffer)
{
    if (buffer.atEnd())
        return std::nullopt;

    auto algorithm = parseHashAlgorithmAdvancingPosition(buffer);
    if (!algorithm)
        return std::nullopt;

    if (!skipExactly(buffer, '-'))
        return std::nullopt;

    auto beginHashValue = buffer.span();
    skipWhile<isBase64OrBase64URLCharacter>(buffer);
    skipExactly(buffer, '=');
    skipExactly(buffer, '=');

    if (buffer.position() == beginHashValue.data())
        return std::nullopt;

    StringView hashValue(beginHashValue.first(buffer.position() - beginHashValue.data()));

    if (auto digest = base64Decode(hashValue))
        return ResourceCryptographicDigest { *algorithm, WTFMove(*digest) };

    if (auto digest = base64URLDecode(hashValue))
        return ResourceCryptographicDigest { *algorithm, WTFMove(*digest) };

    return std::nullopt;
}

std::optional<ResourceCryptographicDigest> parseCryptographicDigest(StringParsingBuffer<UChar>& buffer)
{
    return parseCryptographicDigestImpl(buffer);
}

std::optional<ResourceCryptographicDigest> parseCryptographicDigest(StringParsingBuffer<LChar>& buffer)
{
    return parseCryptographicDigestImpl(buffer);
}

template<typename CharacterType> static std::optional<EncodedResourceCryptographicDigest> parseEncodedCryptographicDigestImpl(StringParsingBuffer<CharacterType>& buffer)
{
    if (buffer.atEnd())
        return std::nullopt;

    auto algorithm = parseHashAlgorithmAdvancingPosition(buffer);
    if (!algorithm)
        return std::nullopt;

    if (!skipExactly(buffer, '-'))
        return std::nullopt;

    auto beginHashValue = buffer.span();
    skipWhile<isBase64OrBase64URLCharacter>(buffer);
    skipExactly(buffer, '=');
    skipExactly(buffer, '=');

    if (buffer.position() == beginHashValue.data())
        return std::nullopt;

    return EncodedResourceCryptographicDigest { *algorithm, beginHashValue.first(buffer.position() - beginHashValue.data()) };
}

std::optional<EncodedResourceCryptographicDigest> parseEncodedCryptographicDigest(StringParsingBuffer<UChar>& buffer)
{
    return parseEncodedCryptographicDigestImpl(buffer);
}

std::optional<EncodedResourceCryptographicDigest> parseEncodedCryptographicDigest(StringParsingBuffer<LChar>& buffer)
{
    return parseEncodedCryptographicDigestImpl(buffer);
}

std::optional<ResourceCryptographicDigest> decodeEncodedResourceCryptographicDigest(const EncodedResourceCryptographicDigest& encodedDigest)
{
    if (auto digest = base64Decode(encodedDigest.digest))
        return ResourceCryptographicDigest { encodedDigest.algorithm, WTFMove(*digest) };

    if (auto digest = base64URLDecode(encodedDigest.digest))
        return ResourceCryptographicDigest { encodedDigest.algorithm, WTFMove(*digest) };

    return std::nullopt;
}

static PAL::CryptoDigest::Algorithm toCryptoDigestAlgorithm(ResourceCryptographicDigest::Algorithm algorithm)
{
    switch (algorithm) {
    case ResourceCryptographicDigest::Algorithm::SHA256:
        return PAL::CryptoDigest::Algorithm::SHA_256;
    case ResourceCryptographicDigest::Algorithm::SHA384:
        return PAL::CryptoDigest::Algorithm::SHA_384;
    case ResourceCryptographicDigest::Algorithm::SHA512:
        return PAL::CryptoDigest::Algorithm::SHA_512;
    }
    ASSERT_NOT_REACHED();
    return PAL::CryptoDigest::Algorithm::SHA_512;
}

ResourceCryptographicDigest cryptographicDigestForBytes(ResourceCryptographicDigest::Algorithm algorithm, std::span<const uint8_t> bytes)
{
    auto cryptoDigest = PAL::CryptoDigest::create(toCryptoDigestAlgorithm(algorithm));
    cryptoDigest->addBytes(bytes);
    return { algorithm, cryptoDigest->computeHash() };
}

ResourceCryptographicDigest cryptographicDigestForSharedBuffer(ResourceCryptographicDigest::Algorithm algorithm, const FragmentedSharedBuffer* buffer)
{
    auto cryptoDigest = PAL::CryptoDigest::create(toCryptoDigestAlgorithm(algorithm));
    if (buffer) {
        buffer->forEachSegment([&](auto segment) {
            cryptoDigest->addBytes(segment);
        });
    }
    return { algorithm, cryptoDigest->computeHash() };
}

}
