/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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
#include "IDBKey.h"

#include "IDBKeyData.h"
#include <JavaScriptCore/ArrayBufferView.h>
#include <JavaScriptCore/JSArrayBuffer.h>
#include <JavaScriptCore/JSArrayBufferView.h>
#include <JavaScriptCore/JSCInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(IDBKey);

using IDBKeyVector = Vector<RefPtr<IDBKey>>;

Ref<IDBKey> IDBKey::createBinary(const ThreadSafeDataBuffer& buffer)
{
    return adoptRef(*new IDBKey(buffer));
}

Ref<IDBKey> IDBKey::createBinary(JSC::JSArrayBuffer& arrayBuffer)
{
    RefPtr buffer = arrayBuffer.impl();
    if (buffer && buffer->isDetached())
        return createInvalid();
    return adoptRef(*new IDBKey(ThreadSafeDataBuffer::copyData(buffer->span())));
}

Ref<IDBKey> IDBKey::createBinary(JSC::JSArrayBufferView& arrayBufferView)
{
    if (arrayBufferView.isDetached())
        return createInvalid();
    auto bufferView = arrayBufferView.possiblySharedImpl();
    if (!bufferView)
        return createInvalid();
    return adoptRef(*new IDBKey(ThreadSafeDataBuffer::copyData(bufferView->span())));
}

IDBKey::IDBKey(IndexedDB::KeyType type, double number)
    : m_type(type)
    , m_value(number)
    , m_sizeEstimate(OverheadSize + sizeof(double))
{
}

IDBKey::IDBKey(const String& value)
    : m_type(IndexedDB::KeyType::String)
    , m_value(value)
    , m_sizeEstimate(OverheadSize + value.length() * sizeof(UChar))
{
}

IDBKey::IDBKey(const IDBKeyVector& keyArray, size_t arraySize)
    : m_type(IndexedDB::KeyType::Array)
    , m_value(keyArray)
    , m_sizeEstimate(OverheadSize + arraySize)
{
}

IDBKey::IDBKey(const ThreadSafeDataBuffer& buffer)
    : m_type(IndexedDB::KeyType::Binary)
    , m_value(buffer)
    , m_sizeEstimate(OverheadSize + buffer.size())
{
}

IDBKey::~IDBKey() = default;

bool IDBKey::isValid() const
{
    if (m_type == IndexedDB::KeyType::Invalid)
        return false;

    if (m_type == IndexedDB::KeyType::Array) {
        for (auto& key : std::get<IDBKeyVector>(m_value)) {
            if (!key->isValid())
                return false;
        }
    }

    return true;
}

int IDBKey::compare(const IDBKey& other) const
{
    if (m_type != other.m_type)
        return m_type > other.m_type ? -1 : 1;

    switch (m_type) {
    case IndexedDB::KeyType::Array: {
        auto& array = std::get<IDBKeyVector>(m_value);
        auto& otherArray = std::get<IDBKeyVector>(other.m_value);
        for (size_t i = 0; i < array.size() && i < otherArray.size(); ++i) {
            if (int result = array[i]->compare(*otherArray[i]))
                return result;
        }
        if (array.size() < otherArray.size())
            return -1;
        if (array.size() > otherArray.size())
            return 1;
        return 0;
    }
    case IndexedDB::KeyType::Binary:
        return compareBinaryKeyData(std::get<ThreadSafeDataBuffer>(m_value), std::get<ThreadSafeDataBuffer>(other.m_value));
    case IndexedDB::KeyType::String:
        return -codePointCompare(std::get<String>(other.m_value), std::get<String>(m_value));
    case IndexedDB::KeyType::Date:
    case IndexedDB::KeyType::Number: {
        auto number = std::get<double>(m_value);
        auto otherNumber = std::get<double>(other.m_value);
        return (number < otherNumber) ? -1 : ((number > otherNumber) ? 1 : 0);
    }
    case IndexedDB::KeyType::Invalid:
    case IndexedDB::KeyType::Min:
    case IndexedDB::KeyType::Max:
        ASSERT_NOT_REACHED();
        return 0;
    }

    ASSERT_NOT_REACHED();
    return 0;
}

bool IDBKey::isLessThan(const IDBKey& other) const
{
    return compare(other) == -1;
}

bool IDBKey::isEqual(const IDBKey& other) const
{
    return !compare(other);
}

#if !LOG_DISABLED
String IDBKey::loggingString() const
{
    return IDBKeyData(this).loggingString();
}
#endif

} // namespace WebCore
