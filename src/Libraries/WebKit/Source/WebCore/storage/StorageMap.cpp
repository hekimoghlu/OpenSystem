/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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
#include "StorageMap.h"

#include <wtf/CheckedArithmetic.h>
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(StorageMap);

StorageMap::StorageMap(unsigned quota)
    : m_impl(Impl::create())
    , m_quotaSize(quota) // quota measured in bytes
{
}

void StorageMap::invalidateIterator()
{
    m_impl->iterator = m_impl->map.end();
    m_impl->iteratorIndex = UINT_MAX;
}

void StorageMap::setIteratorToIndex(unsigned index)
{
    // FIXME: Once we have bidirectional iterators for HashMap we can be more intelligent about this.
    // The requested index will be closest to begin(), our current iterator, or end(), and we
    // can take the shortest route.
    // Until that mechanism is available, we'll always increment our iterator from begin() or current.

    if (m_impl->iteratorIndex == index)
        return;

    if (index < m_impl->iteratorIndex) {
        m_impl->iteratorIndex = 0;
        m_impl->iterator = m_impl->map.begin();
        ASSERT(m_impl->iterator != m_impl->map.end());
    }

    while (m_impl->iteratorIndex < index) {
        ++m_impl->iteratorIndex;
        ++m_impl->iterator;
        ASSERT(m_impl->iterator != m_impl->map.end());
    }
}

unsigned StorageMap::length() const
{
    return m_impl->map.size();
}

String StorageMap::key(unsigned index)
{
    if (index >= length())
        return String();

    setIteratorToIndex(index);
    return m_impl->iterator->key;
}

String StorageMap::getItem(const String& key) const
{
    return m_impl->map.get(key);
}

void StorageMap::setItem(const String& key, const String& value, String& oldValue, bool& quotaException)
{
    ASSERT(!value.isNull());

    quotaException = false;
    CheckedUint32 newSize = m_impl->currentSize;
    auto iter = m_impl->map.find(key);
    if (iter != m_impl->map.end()) {
        oldValue = iter->value;
        newSize -= oldValue.sizeInBytes();
    } else {
        oldValue = nullString();
        newSize += key.sizeInBytes();
    }
    newSize += value.sizeInBytes();
    if (m_quotaSize != noQuota && (newSize.hasOverflowed() || newSize > m_quotaSize)) {
        quotaException = true;
        return;
    }

    // Implement copy-on-write semantics.
    if (m_impl->refCount() > 1)
        m_impl = m_impl->copy();

    m_impl->map.set(key, value);
    m_impl->currentSize = newSize;
    invalidateIterator();
}

void StorageMap::setItemIgnoringQuota(const String& key, const String& value)
{
    SetForScope quotaSizeChange(m_quotaSize, noQuota);

    String oldValue;
    bool quotaException;
    setItem(key, value, oldValue, quotaException);
    ASSERT(!quotaException);
}

void StorageMap::removeItem(const String& key, String& oldValue)
{
    oldValue = nullString();
    CheckedUint32 newSize = m_impl->currentSize;
    auto iter = m_impl->map.find(key);
    if (iter == m_impl->map.end())
        return;
    oldValue = iter->value;
    newSize = newSize - iter->key.sizeInBytes() - oldValue.sizeInBytes();

    if (m_impl->hasOneRef())
        m_impl->map.remove(iter);
    else {
        // Implement copy-on-write semantics.
        m_impl = m_impl->copy();
        m_impl->map.remove(key);
    }

    m_impl->currentSize = newSize;
    invalidateIterator();
}

void StorageMap::clear()
{
    if (m_impl->refCount() > 1 && length()) {
        m_impl = Impl::create();
        return;
    }
    m_impl->map.clear();
    m_impl->currentSize = 0;
    invalidateIterator();
}

bool StorageMap::contains(const String& key) const
{
    return m_impl->map.contains(key);
}

void StorageMap::importItems(HashMap<String, String>&& items)
{
    RELEASE_ASSERT(m_impl->map.isEmpty());
    RELEASE_ASSERT(!m_impl->currentSize);

    CheckedUint32 newSize;
    for (auto& [key, value] : items) {
        newSize += key.sizeInBytes();
        newSize += value.sizeInBytes();
    }

    m_impl->map = WTFMove(items);
    m_impl->currentSize = newSize;
}

Ref<StorageMap::Impl> StorageMap::Impl::copy() const
{
    auto copy = Impl::create();
    copy->map = map;
    copy->currentSize = currentSize;
    return copy;
}

}
