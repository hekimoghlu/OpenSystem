/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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

#include "IndexedDB.h"
#include "ThreadSafeDataBuffer.h"
#include <variant>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class JSArrayBuffer;
class JSArrayBufferView;
}

namespace WebCore {

class IDBKey : public RefCounted<IDBKey> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(IDBKey);
public:
    static Ref<IDBKey> createInvalid()
    {
        return adoptRef(*new IDBKey());
    }

    static Ref<IDBKey> createNumber(double number)
    {
        return adoptRef(*new IDBKey(IndexedDB::KeyType::Number, number));
    }

    static Ref<IDBKey> createString(const String& string)
    {
        return adoptRef(*new IDBKey(string));
    }

    static Ref<IDBKey> createDate(double date)
    {
        return adoptRef(*new IDBKey(IndexedDB::KeyType::Date, date));
    }

    static Ref<IDBKey> createMultiEntryArray(const Vector<RefPtr<IDBKey>>& array)
    {
        Vector<RefPtr<IDBKey>> result;

        size_t sizeEstimate = 0;
        for (auto& key : array) {
            if (!key->isValid())
                continue;

            bool skip = false;
            for (auto& resultKey : result) {
                if (key->isEqual(*resultKey)) {
                    skip = true;
                    break;
                }
            }
            if (!skip) {
                result.append(key);
                sizeEstimate += key->m_sizeEstimate;
            }
        }
        auto idbKey = adoptRef(*new IDBKey(result, sizeEstimate));
        ASSERT(idbKey->isValid());
        return idbKey;
    }

    static Ref<IDBKey> createArray(const Vector<RefPtr<IDBKey>>& array)
    {
        size_t sizeEstimate = 0;
        for (auto& key : array)
            sizeEstimate += key->m_sizeEstimate;

        return adoptRef(*new IDBKey(array, sizeEstimate));
    }

    static Ref<IDBKey> createBinary(const ThreadSafeDataBuffer&);
    static Ref<IDBKey> createBinary(JSC::JSArrayBuffer&);
    static Ref<IDBKey> createBinary(JSC::JSArrayBufferView&);

    WEBCORE_EXPORT ~IDBKey();

    IndexedDB::KeyType type() const { return m_type; }
    WEBCORE_EXPORT bool isValid() const;

    const Vector<RefPtr<IDBKey>>& array() const
    {
        ASSERT(m_type == IndexedDB::KeyType::Array);
        return std::get<Vector<RefPtr<IDBKey>>>(m_value);
    }

    const String& string() const
    {
        ASSERT(m_type == IndexedDB::KeyType::String);
        return std::get<String>(m_value);
    }

    double date() const
    {
        ASSERT(m_type == IndexedDB::KeyType::Date);
        return std::get<double>(m_value);
    }

    double number() const
    {
        ASSERT(m_type == IndexedDB::KeyType::Number);
        return std::get<double>(m_value);
    }

    const ThreadSafeDataBuffer& binary() const
    {
        ASSERT(m_type == IndexedDB::KeyType::Binary);
        return std::get<ThreadSafeDataBuffer>(m_value);
    }

    int compare(const IDBKey& other) const;
    bool isLessThan(const IDBKey& other) const;
    bool isEqual(const IDBKey& other) const;

    size_t sizeEstimate() const { return m_sizeEstimate; }

    using RefCounted<IDBKey>::ref;
    using RefCounted<IDBKey>::deref;

#if !LOG_DISABLED
    String loggingString() const;
#endif

private:
    IDBKey()
        : m_type(IndexedDB::KeyType::Invalid)
        , m_sizeEstimate(OverheadSize)
    {
    }

    IDBKey(IndexedDB::KeyType, double number);
    explicit IDBKey(const String& value);
    IDBKey(const Vector<RefPtr<IDBKey>>& keyArray, size_t arraySize);
    explicit IDBKey(const ThreadSafeDataBuffer&);

    const IndexedDB::KeyType m_type;
    std::variant<Vector<RefPtr<IDBKey>>, String, double, ThreadSafeDataBuffer> m_value;

    const size_t m_sizeEstimate;

    // Very rough estimate of minimum key size overhead.
    enum { OverheadSize = 16 };
};

inline int compareBinaryKeyData(const Vector<uint8_t>& a, const Vector<uint8_t>& b)
{
    size_t length = std::min(a.size(), b.size());

    for (size_t i = 0; i < length; ++i) {
        if (a[i] > b[i])
            return 1;
        if (a[i] < b[i])
            return -1;
    }

    if (a.size() == b.size())
        return 0;

    if (a.size() > b.size())
        return 1;

    return -1;
}

inline int compareBinaryKeyData(const ThreadSafeDataBuffer& a, const ThreadSafeDataBuffer& b)
{
    auto* aData = a.data();
    auto* bData = b.data();

    // Covers the cases where both pointers are null as well as both pointing to the same buffer.
    if (aData == bData)
        return 0;

    if (aData && !bData)
        return 1;
    if (!aData && bData)
        return -1;

    return compareBinaryKeyData(*aData, *bData);
}

} // namespace WebCore
