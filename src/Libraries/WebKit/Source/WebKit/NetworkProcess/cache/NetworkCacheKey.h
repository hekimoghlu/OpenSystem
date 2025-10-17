/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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

#include "NetworkCacheData.h"
#include <wtf/CrossThreadCopier.h>
#include <wtf/SHA1.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/WTFString.h>

namespace WTF::Persistence {
template<typename> struct Coder;
}

namespace WebKit::NetworkCache {

struct DataKey {
    String partition;
    String type;
    SHA1::Digest identifier;

    template <class Encoder> void encodeForPersistence(Encoder& encoder) const
    {
        encoder << partition << type << identifier;
    }

    template <class Decoder> static WARN_UNUSED_RETURN bool decodeForPersistence(Decoder& decoder, DataKey& dataKey)
    {
        return decoder.decode(dataKey.partition) && decoder.decode(dataKey.type) && decoder.decode(dataKey.identifier);
    }
};

class Key {
public:
    typedef SHA1::Digest HashType;

    Key() { }
    Key(const Key&);
    Key(Key&&) = default;
    Key(const String& partition, const String& type, const String& range, const String& identifier, const Salt&);
    Key(const DataKey&, const Salt&);

    Key& operator=(const Key&);
    Key& operator=(Key&&) = default;

    Key(WTF::HashTableDeletedValueType);
    bool isHashTableDeletedValue() const { return m_identifier.isHashTableDeletedValue(); }

    bool isNull() const { return m_identifier.isNull(); }

    const String& partition() const { return m_partition; }
    const String& identifier() const { return m_identifier; }
    const String& type() const { return m_type; }
    const String& range() const { return m_range; }

    const HashType& hash() const { return m_hash; }
    const HashType& partitionHash() const { return m_partitionHash; }

    static bool stringToHash(const String&, HashType&);

    static size_t hashStringLength() { return 2 * sizeof(m_hash); }
    String hashAsString() const { return hashAsString(m_hash); }
    String partitionHashAsString() const { return hashAsString(m_partitionHash); }

    bool operator==(const Key&) const;

    static String partitionToPartitionHashAsString(const String& partition, const Salt&);

    Key isolatedCopy() && { return {
        crossThreadCopy(WTFMove(m_partition)),
        crossThreadCopy(WTFMove(m_type)),
        crossThreadCopy(WTFMove(m_identifier)),
        crossThreadCopy(WTFMove(m_range)),
        m_hash,
        m_partitionHash
    }; }

    Key isolatedCopy() const & { return {
        crossThreadCopy(m_partition),
        crossThreadCopy(m_type),
        crossThreadCopy(m_identifier),
        crossThreadCopy(m_range),
        m_hash,
        m_partitionHash
    }; }

private:
    friend struct WTF::Persistence::Coder<Key>;
    static String hashAsString(const HashType&);
    HashType computeHash(const Salt&) const;
    HashType computePartitionHash(const Salt&) const;
    static HashType partitionToPartitionHash(const String& partition, const Salt&);
    Key(String&& partition, String&& type, String&& identifier, String&& range, HashType hash, HashType partitionHash)
        : m_partition(WTFMove(partition))
        , m_type(WTFMove(type))
        , m_identifier(WTFMove(identifier))
        , m_range(WTFMove(range))
        , m_hash(hash)
        , m_partitionHash(partitionHash) { }

    String m_partition;
    String m_type;
    String m_identifier;
    String m_range;
    HashType m_hash;
    HashType m_partitionHash;
};

}

namespace WTF {

struct NetworkCacheKeyHash {
    static unsigned hash(const WebKit::NetworkCache::Key& key)
    {
        static_assert(SHA1::hashSize >= sizeof(unsigned), "Hash size must be greater than sizeof(unsigned)");
        return reinterpretCastSpanStartTo<const unsigned>(std::span { key.hash() });
    }

    static bool equal(const WebKit::NetworkCache::Key& a, const WebKit::NetworkCache::Key& b)
    {
        return a == b;
    }

    static const bool safeToCompareToEmptyOrDeleted = false;
};

template<typename T> struct DefaultHash;
template<> struct DefaultHash<WebKit::NetworkCache::Key> : NetworkCacheKeyHash { };

template<> struct HashTraits<WebKit::NetworkCache::Key> : SimpleClassHashTraits<WebKit::NetworkCache::Key> {
    static const bool emptyValueIsZero = false;

    static const bool hasIsEmptyValueFunction = true;
    static bool isEmptyValue(const WebKit::NetworkCache::Key& key) { return key.isNull(); }
};

} // namespace WTF
