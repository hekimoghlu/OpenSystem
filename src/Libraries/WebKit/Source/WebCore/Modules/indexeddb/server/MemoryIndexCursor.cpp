/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#include "MemoryIndexCursor.h"

#include "IDBCursorInfo.h"
#include "IDBGetResult.h"
#include "IndexValueStore.h"
#include "Logging.h"
#include "MemoryCursor.h"
#include "MemoryIndex.h"
#include "MemoryObjectStore.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace IDBServer {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MemoryIndexCursor);

Ref<MemoryIndexCursor> MemoryIndexCursor::create(MemoryIndex& index, const IDBCursorInfo& info, MemoryBackingStoreTransaction& transaction)
{
    return adoptRef(*new MemoryIndexCursor(index, info, transaction));
}

MemoryIndexCursor::MemoryIndexCursor(MemoryIndex& index, const IDBCursorInfo& cursorInfo, MemoryBackingStoreTransaction& transaction)
    : MemoryCursor(cursorInfo, transaction)
    , m_index(index)
{
    LOG(IndexedDB, "MemoryIndexCursor::MemoryIndexCursor %s", cursorInfo.range().loggingString().utf8().data());

    auto* valueStore = index.valueStore();
    if (!valueStore)
        return;

    if (info().isDirectionForward())
        m_currentIterator = valueStore->find(info().range().lowerKey, info().range().lowerOpen);
    else
        m_currentIterator = valueStore->reverseFind(info().range().upperKey, info().duplicity(), info().range().upperOpen);

    if (m_currentIterator.isValid() && info().range().containsKey(m_currentIterator.key())) {
        m_currentKey = m_currentIterator.key();
        m_currentPrimaryKey = m_currentIterator.primaryKey();
        index.cursorDidBecomeClean(*this);
    } else
        m_currentIterator.invalidate();
}

MemoryIndexCursor::~MemoryIndexCursor() = default;

void MemoryIndexCursor::currentData(IDBGetResult& getResult)
{
    if (!m_currentIterator.isValid()) {
        getResult = { };
        return;
    }

    if (info().cursorType() == IndexedDB::CursorType::KeyOnly)
        getResult = { m_currentKey, m_currentPrimaryKey };
    else {
        IDBValue value = { m_index->protectedObjectStore()->valueForKey(m_currentPrimaryKey), { }, { } };
        getResult = { m_currentKey, m_currentPrimaryKey, WTFMove(value), m_index->protectedObjectStore()->info().keyPath() };
    }
}

void MemoryIndexCursor::iterate(const IDBKeyData& key, const IDBKeyData& primaryKey, uint32_t count, IDBGetResult& getResult)
{
    LOG(IndexedDB, "MemoryIndexCursor::iterate to key %s, %u count", key.loggingString().utf8().data(), count);

#ifndef NDEBUG
    if (primaryKey.isValid())
        ASSERT(key.isValid());
#endif

    Ref index = m_index.get();
    if (key.isValid()) {
        // Cannot iterate by both a count and to a key
        ASSERT(!count);

        auto* valueStore = index->valueStore();
        if (!valueStore) {
            m_currentKey = { };
            m_currentPrimaryKey = { };
            getResult = { };
            return;
        }

        if (primaryKey.isValid()) {
            if (info().isDirectionForward())
                m_currentIterator = valueStore->find(key, primaryKey);
            else
                m_currentIterator = valueStore->reverseFind(key, primaryKey, info().duplicity());
        } else {
            if (info().isDirectionForward())
                m_currentIterator = valueStore->find(key);
            else
                m_currentIterator = valueStore->reverseFind(key, info().duplicity());
        }

        if (m_currentIterator.isValid() && !info().range().containsKey(m_currentIterator.key()))
            m_currentIterator.invalidate();

        if (!m_currentIterator.isValid()) {
            m_currentKey = { };
            m_currentPrimaryKey = { };
            getResult = { };
            return;
        }

        index->cursorDidBecomeClean(*this);

        m_currentKey = m_currentIterator.key();
        m_currentPrimaryKey = m_currentIterator.primaryKey();
        currentData(getResult);

        return;
    }

    // If there was not a valid key argument and no positive count argument
    // that means the default iteration count of "1"
    if (!count)
        count = 1;

    if (!m_currentIterator.isValid()) {
        auto* valueStore = index->valueStore();
        if (!valueStore) {
            m_currentKey = { };
            m_currentPrimaryKey = { };
            getResult = { };
            return;
        }

        switch (info().cursorDirection()) {
        case IndexedDB::CursorDirection::Next:
            m_currentIterator = valueStore->find(m_currentKey, m_currentPrimaryKey);
            break;
        case IndexedDB::CursorDirection::Nextunique:
            m_currentIterator = valueStore->find(m_currentKey, true);
            break;
        case IndexedDB::CursorDirection::Prev:
            m_currentIterator = valueStore->reverseFind(m_currentKey, m_currentPrimaryKey, info().duplicity());
            break;
        case IndexedDB::CursorDirection::Prevunique:
            m_currentIterator = valueStore->reverseFind(m_currentKey, info().duplicity(), true);
            break;
        }

        if (!m_currentIterator.isValid()) {
            m_currentKey = { };
            m_currentPrimaryKey = { };
            getResult = { };
            return;
        }

        index->cursorDidBecomeClean(*this);

        // If we restored the current iterator and it does *not* match the current key/primaryKey,
        // then it is the next record in line and we should consider that an iteration.
        if (m_currentKey != m_currentIterator.key() || m_currentPrimaryKey != m_currentIterator.primaryKey())
            --count;
    }

    ASSERT(m_currentIterator.isValid());

    while (count) {
        if (info().duplicity() == CursorDuplicity::NoDuplicates)
            m_currentIterator.nextIndexEntry();
        else
            ++m_currentIterator;

        if (!m_currentIterator.isValid())
            break;

        --count;
    }

    if (m_currentIterator.isValid() && !info().range().containsKey(m_currentIterator.key()))
        m_currentIterator.invalidate();

    // Not having a valid iterator after finishing any iteration means we've reached the end of the cursor.
    if (!m_currentIterator.isValid()) {
        m_currentKey = { };
        m_currentPrimaryKey = { };
        getResult = { };
        return;
    }

    m_currentKey = m_currentIterator.key();
    m_currentPrimaryKey = m_currentIterator.primaryKey();
    currentData(getResult);
}

Ref<MemoryIndex> MemoryIndexCursor::protectedIndex() const
{
    return m_index.get();
}

void MemoryIndexCursor::indexRecordsAllChanged()
{
    m_currentIterator.invalidate();
    protectedIndex()->cursorDidBecomeDirty(*this);
}

void MemoryIndexCursor::indexValueChanged(const IDBKeyData& key, const IDBKeyData& primaryKey)
{
    if (m_currentKey != key || m_currentPrimaryKey != primaryKey)
        return;

    m_currentIterator.invalidate();
    protectedIndex()->cursorDidBecomeDirty(*this);
}

} // namespace IDBServer
} // namespace WebCore
