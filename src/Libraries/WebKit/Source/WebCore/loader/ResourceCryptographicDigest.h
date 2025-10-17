/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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

#include <type_traits>
#include <wtf/HashTraits.h>
#include <wtf/Hasher.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CachedResource;
class FragmentedSharedBuffer;

struct ResourceCryptographicDigest {
    static constexpr unsigned algorithmCount = 3;
    enum class Algorithm : uint8_t {
        SHA256 = 1 << 0,
        SHA384 = 1 << 1,
        SHA512 = 1 << (algorithmCount - 1),
    };

    // Number of bytes to hold SHA-512 digest
    static constexpr size_t maximumDigestLength = 64;

    Algorithm algorithm;
    Vector<uint8_t> value;

    friend bool operator==(const ResourceCryptographicDigest&, const ResourceCryptographicDigest&) = default;
};

inline void add(Hasher& hasher, const ResourceCryptographicDigest& digest)
{
    add(hasher, digest.algorithm, digest.value);
}

struct EncodedResourceCryptographicDigest {
    using Algorithm = ResourceCryptographicDigest::Algorithm;
    
    Algorithm algorithm;
    String digest;
};

std::optional<ResourceCryptographicDigest> parseCryptographicDigest(StringParsingBuffer<UChar>&);
std::optional<ResourceCryptographicDigest> parseCryptographicDigest(StringParsingBuffer<LChar>&);

std::optional<EncodedResourceCryptographicDigest> parseEncodedCryptographicDigest(StringParsingBuffer<UChar>&);
std::optional<EncodedResourceCryptographicDigest> parseEncodedCryptographicDigest(StringParsingBuffer<LChar>&);

std::optional<ResourceCryptographicDigest> decodeEncodedResourceCryptographicDigest(const EncodedResourceCryptographicDigest&);

ResourceCryptographicDigest cryptographicDigestForSharedBuffer(ResourceCryptographicDigest::Algorithm, const FragmentedSharedBuffer*);
ResourceCryptographicDigest cryptographicDigestForBytes(ResourceCryptographicDigest::Algorithm, std::span<const uint8_t> bytes);

}

namespace WTF {

template<> struct DefaultHash<WebCore::ResourceCryptographicDigest> {
    static unsigned hash(const WebCore::ResourceCryptographicDigest& digest)
    {
        return computeHash(digest);
    }
    static bool equal(const WebCore::ResourceCryptographicDigest& a, const WebCore::ResourceCryptographicDigest& b)
    {
        return a == b;
    }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct HashTraits<WebCore::ResourceCryptographicDigest> : GenericHashTraits<WebCore::ResourceCryptographicDigest> {
    using Algorithm = WebCore::ResourceCryptographicDigest::Algorithm;
    using AlgorithmUnderlyingType = typename std::underlying_type<Algorithm>::type;
    static constexpr auto emptyAlgorithmValue = static_cast<Algorithm>(std::numeric_limits<AlgorithmUnderlyingType>::max());
    static constexpr auto deletedAlgorithmValue = static_cast<Algorithm>(std::numeric_limits<AlgorithmUnderlyingType>::max() - 1);

    static const bool emptyValueIsZero = false;

    static WebCore::ResourceCryptographicDigest emptyValue()
    {
        return { emptyAlgorithmValue, { } };
    }

    static bool isEmptyValue(const WebCore::ResourceCryptographicDigest& value) { return value.algorithm == emptyAlgorithmValue; }

    static void constructDeletedValue(WebCore::ResourceCryptographicDigest& slot)
    {
        slot.algorithm = deletedAlgorithmValue;
    }

    static bool isDeletedValue(const WebCore::ResourceCryptographicDigest& slot)
    {
        return slot.algorithm == deletedAlgorithmValue;
    }
};

}
