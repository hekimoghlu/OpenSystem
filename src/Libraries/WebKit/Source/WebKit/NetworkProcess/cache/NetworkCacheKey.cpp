/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
#include "NetworkCacheKey.h"

#include "NetworkCacheCoders.h"
#include <wtf/ASCIICType.h>
#include <wtf/persistence/PersistentDecoder.h>
#include <wtf/persistence/PersistentEncoder.h>
#include <wtf/text/CString.h>
#include <wtf/text/StringBuilder.h>

namespace WebKit {
namespace NetworkCache {

Key::Key(const Key& o)
    : m_partition(o.m_partition.isolatedCopy())
    , m_type(o.m_type.isolatedCopy())
    , m_identifier(o.m_identifier.isolatedCopy())
    , m_range(o.m_range.isolatedCopy())
    , m_hash(o.m_hash)
    , m_partitionHash(o.m_partitionHash)
{
}

Key::Key(const String& partition, const String& type, const String& range, const String& identifier, const Salt& salt)
    : m_partition(partition)
    , m_type(type)
    , m_identifier(identifier)
    , m_range(range)
    , m_hash(computeHash(salt))
    , m_partitionHash(computePartitionHash(salt))
{
}

Key::Key(WTF::HashTableDeletedValueType)
    : m_identifier(WTF::HashTableDeletedValue)
{
}

Key::Key(const DataKey& dataKey, const Salt& salt)
    : m_partition(dataKey.partition)
    , m_type(dataKey.type)
    , m_identifier(hashAsString(dataKey.identifier))
    , m_hash(computeHash(salt))
    , m_partitionHash(computePartitionHash(salt))
{
}

Key& Key::operator=(const Key& other)
{
    m_partition = other.m_partition.isolatedCopy();
    m_type = other.m_type.isolatedCopy();
    m_identifier = other.m_identifier.isolatedCopy();
    m_range = other.m_range.isolatedCopy();
    m_hash = other.m_hash;
    m_partitionHash = other.m_partitionHash;
    return *this;
}

static void hashString(SHA1& sha1, const String& string)
{
    if (string.isNull())
        return;

    sha1.addUTF8Bytes(string);
}

Key::HashType Key::computeHash(const Salt& salt) const
{
    // We don't really need a cryptographic hash. The key is always verified against the entry header.
    // SHA1 just happens to be suitably sized, fast and available.
    SHA1 sha1;
    sha1.addBytes(salt);

    hashString(sha1, m_partition);
    hashString(sha1, m_type);
    hashString(sha1, m_identifier);
    hashString(sha1, m_range);

    SHA1::Digest hash;
    sha1.computeHash(hash);
    return hash;
}

String Key::partitionToPartitionHashAsString(const String& partition, const Salt& salt)
{
    return hashAsString(partitionToPartitionHash(partition, salt));
}

Key::HashType Key::computePartitionHash(const Salt& salt) const
{
    return partitionToPartitionHash(m_partition, salt);
}

Key::HashType Key::partitionToPartitionHash(const String& partition, const Salt& salt)
{
    SHA1 sha1;
    sha1.addBytes(salt);

    hashString(sha1, partition);

    SHA1::Digest hash;
    sha1.computeHash(hash);
    return hash;
}

String Key::hashAsString(const HashType& hash)
{
    StringBuilder builder;
    builder.reserveCapacity(hashStringLength());
    for (auto byte : hash) {
        builder.append(upperNibbleToASCIIHexDigit(byte));
        builder.append(lowerNibbleToASCIIHexDigit(byte));
    }
    return builder.toString();
}

template <typename CharType> bool hexDigitsToHash(std::span<const CharType> characters, Key::HashType& hash)
{
    for (unsigned i = 0; i < sizeof(hash); ++i) {
        auto high = characters[2 * i];
        auto low = characters[2 * i + 1];
        if (!isASCIIHexDigit(high) || !isASCIIHexDigit(low))
            return false;
        hash[i] = toASCIIHexValue(high, low);
    }
    return true;
}

bool Key::stringToHash(const String& string, HashType& hash)
{
    if (string.length() != hashStringLength())
        return false;
    if (string.is8Bit())
        return hexDigitsToHash(string.span8(), hash);
    return hexDigitsToHash(string.span16(), hash);
}

bool Key::operator==(const Key& other) const
{
    return m_hash == other.m_hash && m_partition == other.m_partition && m_type == other.m_type && m_identifier == other.m_identifier && m_range == other.m_range;
}

}
}
