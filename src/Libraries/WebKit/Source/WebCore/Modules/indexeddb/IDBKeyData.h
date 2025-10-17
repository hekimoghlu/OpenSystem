/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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

#include "IDBKey.h"
#include <variant>
#include <wtf/Hasher.h>
#include <wtf/StdSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class KeyedDecoder;
class KeyedEncoder;

class IDBKeyData {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(IDBKeyData, WEBCORE_EXPORT);
public:
    struct Date {
        double value { 0.0 };
        Date isolatedCopy() const { return { value }; }
    };
    struct Min { Min isolatedCopy() const { return { }; } };
    struct Max { Max isolatedCopy() const { return { }; } };
    struct Invalid { Invalid isolatedCopy() const { return { }; } };
    using ValueVariant = std::variant<std::nullptr_t, Invalid, Vector<IDBKeyData>, String, double, Date, ThreadSafeDataBuffer, Min, Max>;

    enum IsolatedCopyTag { IsolatedCopy };

    IDBKeyData() = default;
    IDBKeyData(ValueVariant&& value)
        : m_value(WTFMove(value)) { }
    IDBKeyData(const IDBKeyData&, IsolatedCopyTag);
    WEBCORE_EXPORT IDBKeyData(const IDBKey*);

    static IDBKeyData minimum()
    {
        IDBKeyData result;
        result.m_value = Min { };
        return result;
    }

    static IDBKeyData maximum()
    {
        IDBKeyData result;
        result.m_value = Max { };
        return result;
    }

    WEBCORE_EXPORT RefPtr<IDBKey> maybeCreateIDBKey() const;

    WEBCORE_EXPORT IDBKeyData isolatedCopy() const;

    WEBCORE_EXPORT void encode(KeyedEncoder&) const;
    WEBCORE_EXPORT static WARN_UNUSED_RETURN bool decode(KeyedDecoder&, IDBKeyData&);

    // compare() has the same semantics as strcmp().
    //   - Returns negative if this IDBKeyData is less than other.
    //   - Returns positive if this IDBKeyData is greater than other.
    //   - Returns zero if this IDBKeyData is equal to other.
    WEBCORE_EXPORT int compare(const IDBKeyData& other) const;

    void setArrayValue(const Vector<IDBKeyData>&);
    void setBinaryValue(const ThreadSafeDataBuffer&);
    void setStringValue(const String&);
    void setDateValue(double);
    WEBCORE_EXPORT void setNumberValue(double);
    
#if !LOG_DISABLED
    WEBCORE_EXPORT String loggingString() const;
#endif

    bool isNull() const { return std::holds_alternative<std::nullptr_t>(m_value); }
    bool isValid() const;
    WEBCORE_EXPORT static bool isValidValue(const ValueVariant&);
    IndexedDB::KeyType type() const;

    bool operator<(const IDBKeyData&) const;
    bool operator>(const IDBKeyData& other) const
    {
        return !(*this < other) && !(*this == other);
    }

    bool operator<=(const IDBKeyData& other) const
    {
        return !(*this > other);
    }

    bool operator>=(const IDBKeyData& other) const
    {
        return !(*this < other);
    }

    bool operator==(const IDBKeyData& other) const;

    String string() const
    {
        return std::get<String>(m_value);
    }

    double date() const
    {
        return std::get<Date>(m_value).value;
    }

    double number() const
    {
        return std::get<double>(m_value);
    }

    const ThreadSafeDataBuffer& binary() const
    {
        return std::get<ThreadSafeDataBuffer>(m_value);
    }

    const Vector<IDBKeyData>& array() const
    {
        return std::get<Vector<IDBKeyData>>(m_value);
    }

    size_t size() const;

    const ValueVariant& value() const { return m_value; };

private:
    friend struct IDBKeyDataHashTraits;

    bool m_isDeletedValue { false };
    ValueVariant m_value;
};

inline void add(Hasher& hasher, const IDBKeyData& keyData)
{
    add(hasher, keyData.type());
    add(hasher, keyData.isNull());
    switch (keyData.type()) {
    case IndexedDB::KeyType::Invalid:
    case IndexedDB::KeyType::Max:
    case IndexedDB::KeyType::Min:
        break;
    case IndexedDB::KeyType::Number:
        add(hasher, keyData.number());
        break;
    case IndexedDB::KeyType::Date:
        add(hasher, keyData.date());
        break;
    case IndexedDB::KeyType::String:
        add(hasher, keyData.string());
        break;
    case IndexedDB::KeyType::Binary:
        add(hasher, keyData.binary());
        break;
    case IndexedDB::KeyType::Array:
        add(hasher, keyData.array());
        break;
    }
}

struct IDBKeyDataHash {
    static unsigned hash(const IDBKeyData& a) { return computeHash(a); }
    static bool equal(const IDBKeyData& a, const IDBKeyData& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

struct IDBKeyDataHashTraits : public WTF::CustomHashTraits<IDBKeyData> {
    static const bool emptyValueIsZero = false;
    static const bool hasIsEmptyValueFunction = true;

    static void constructDeletedValue(IDBKeyData& key) { key.m_isDeletedValue = true; }
    static bool isDeletedValue(const IDBKeyData& key) { return key.m_isDeletedValue; }

    static IDBKeyData emptyValue()
    {
        return IDBKeyData();
    }

    static bool isEmptyValue(const IDBKeyData& key)
    {
        return key.isNull();
    }
};

using IDBKeyDataSet = StdSet<IDBKeyData>;

} // namespace WebCore
