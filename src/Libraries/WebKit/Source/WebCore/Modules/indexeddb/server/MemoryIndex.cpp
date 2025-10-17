/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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
#include "MemoryIndex.h"

#include "IDBError.h"
#include "IDBGetAllResult.h"
#include "IDBGetResult.h"
#include "IDBKeyRangeData.h"
#include "IndexKey.h"
#include "Logging.h"
#include "MemoryBackingStoreTransaction.h"
#include "MemoryIndexCursor.h"
#include "MemoryObjectStore.h"
#include "ThreadSafeDataBuffer.h"

namespace WebCore {
namespace IDBServer {

Ref<MemoryIndex> MemoryIndex::create(const IDBIndexInfo& info, MemoryObjectStore& objectStore)
{
    return adoptRef(*new MemoryIndex(info, objectStore));
}

MemoryIndex::MemoryIndex(const IDBIndexInfo& info, MemoryObjectStore& objectStore)
    : m_info(info)
    , m_objectStore(objectStore)
{
}

MemoryIndex::~MemoryIndex() = default;

WeakPtr<MemoryObjectStore> MemoryIndex::objectStore()
{
    return m_objectStore;
}

RefPtr<MemoryObjectStore> MemoryIndex::protectedObjectStore()
{
    return m_objectStore.get();
}

void MemoryIndex::cursorDidBecomeClean(MemoryIndexCursor& cursor)
{
    m_cleanCursors.add(cursor);
}

void MemoryIndex::cursorDidBecomeDirty(MemoryIndexCursor& cursor)
{
    m_cleanCursors.remove(cursor);
}

void MemoryIndex::objectStoreCleared()
{
    auto transaction = m_objectStore->writeTransaction();
    ASSERT(transaction);

    transaction->indexCleared(*this, WTFMove(m_records));

    notifyCursorsOfAllRecordsChanged();
}

void MemoryIndex::notifyCursorsOfValueChange(const IDBKeyData& indexKey, const IDBKeyData& primaryKey)
{
    for (WeakPtr cursor : copyToVector(m_cleanCursors)) {
        if (RefPtr protectedCusor = cursor.get())
            protectedCusor->indexValueChanged(indexKey, primaryKey);
    }
}

void MemoryIndex::notifyCursorsOfAllRecordsChanged()
{
    for (WeakPtr cursor : copyToVector(m_cleanCursors)) {
        if (RefPtr protectedCusor = cursor.get())
            protectedCusor->indexRecordsAllChanged();
    }

    ASSERT(!m_cleanCursors.computeSize());
}

void MemoryIndex::clearIndexValueStore()
{
    ASSERT(m_objectStore->writeTransaction());
    ASSERT(m_objectStore->writeTransaction()->isAborting());

    m_records = nullptr;
}

void MemoryIndex::replaceIndexValueStore(std::unique_ptr<IndexValueStore>&& valueStore)
{
    ASSERT(m_objectStore->writeTransaction());
    ASSERT(m_objectStore->writeTransaction()->isAborting());

    m_records = WTFMove(valueStore);
}

IDBGetResult MemoryIndex::getResultForKeyRange(IndexedDB::IndexRecordType type, const IDBKeyRangeData& range) const
{
    LOG(IndexedDB, "MemoryIndex::getResultForKeyRange - %s", range.loggingString().utf8().data());

    if (!m_records)
        return { };

    IDBKeyData keyToLookFor;
    if (range.isExactlyOneKey())
        keyToLookFor = range.lowerKey;
    else
        keyToLookFor = m_records->lowestKeyWithRecordInRange(range);

    if (keyToLookFor.isNull())
        return { };

    const IDBKeyData* keyValue = m_records->lowestValueForKey(keyToLookFor);

    if (!keyValue)
        return { };

    return type == IndexedDB::IndexRecordType::Key ? IDBGetResult(*keyValue) : IDBGetResult(*keyValue, m_objectStore->valueForKeyRange(*keyValue), m_objectStore->info().keyPath());
}

uint64_t MemoryIndex::countForKeyRange(const IDBKeyRangeData& inRange)
{
    LOG(IndexedDB, "MemoryIndex::countForKeyRange");

    if (!m_records)
        return 0;

    uint64_t count = 0;
    IDBKeyRangeData range = inRange;
    while (true) {
        auto key = m_records->lowestKeyWithRecordInRange(range);
        if (key.isNull())
            break;

        count += m_records->countForKey(key);

        range.lowerKey = key;
        range.lowerOpen = true;
    }

    return count;
}

void MemoryIndex::getAllRecords(const IDBKeyRangeData& keyRangeData, std::optional<uint32_t> count, IndexedDB::GetAllType type, IDBGetAllResult& result) const
{
    LOG(IndexedDB, "MemoryIndex::getAllRecords");

    result = { type, m_objectStore->info().keyPath() };

    if (!m_records)
        return;

    uint32_t targetCount;
    if (count && count.value())
        targetCount = count.value();
    else
        targetCount = std::numeric_limits<uint32_t>::max();

    IDBKeyRangeData range = keyRangeData;
    uint32_t currentCount = 0;
    while (currentCount < targetCount) {
        auto key = m_records->lowestKeyWithRecordInRange(range);
        if (key.isNull())
            return;

        range.lowerKey = key;
        range.lowerOpen = true;

        auto allValues = m_records->allValuesForKey(key, targetCount - currentCount);
        for (auto& keyValue : allValues) {
            result.addKey(IDBKeyData(keyValue));
            if (type == IndexedDB::GetAllType::Values)
                result.addValue(m_objectStore->valueForKeyRange(keyValue));
        }

        currentCount += allValues.size();
    }
}


IDBError MemoryIndex::putIndexKey(const IDBKeyData& valueKey, const IndexKey& indexKey)
{
    LOG(IndexedDB, "MemoryIndex::provisionalPutIndexKey");

    if (!m_records) {
        m_records = makeUnique<IndexValueStore>(m_info.unique());
        notifyCursorsOfAllRecordsChanged();
    }

    if (!m_info.multiEntry()) {
        IDBKeyData key = indexKey.asOneKey();
        IDBError result = m_records->addRecord(key, valueKey);
        notifyCursorsOfValueChange(key, valueKey);
        return result;
    }

    Vector<IDBKeyData> keys = indexKey.multiEntry();

    if (m_info.unique()) {
        for (auto& key : keys) {
            if (m_records->contains(key))
                return IDBError(ExceptionCode::ConstraintError);
        }
    }

    for (auto& key : keys) {
        auto error = m_records->addRecord(key, valueKey);
        ASSERT_UNUSED(error, error.isNull());
        notifyCursorsOfValueChange(key, valueKey);
    }

    return IDBError { };
}

void MemoryIndex::removeRecord(const IDBKeyData& valueKey, const IndexKey& indexKey)
{
    LOG(IndexedDB, "MemoryIndex::removeRecord");

    ASSERT(m_records);

    if (!m_info.multiEntry()) {
        IDBKeyData key = indexKey.asOneKey();
        m_records->removeRecord(key, valueKey);
        notifyCursorsOfValueChange(key, valueKey);
        return;
    }

    Vector<IDBKeyData> keys = indexKey.multiEntry();
    for (auto& key : keys) {
        m_records->removeRecord(key, valueKey);
        notifyCursorsOfValueChange(key, valueKey);
    }
}

void MemoryIndex::removeEntriesWithValueKey(const IDBKeyData& valueKey)
{
    LOG(IndexedDB, "MemoryIndex::removeEntriesWithValueKey");

    if (!m_records)
        return;

    m_records->removeEntriesWithValueKey(*this, valueKey);
}

MemoryIndexCursor* MemoryIndex::maybeOpenCursor(const IDBCursorInfo& info, MemoryBackingStoreTransaction& transaction)
{
    if (transaction.isWriting()) {
        RefPtr objectStore = m_objectStore.get();
        if (!objectStore)
            return nullptr;

        if (objectStore->writeTransaction() != &transaction)
            return nullptr;
    }

    auto result = m_cursors.add(info.identifier(), nullptr);
    if (!result.isNewEntry)
        return nullptr;

    result.iterator->value = MemoryIndexCursor::create(*this, info, transaction);
    return result.iterator->value.get();
}

void MemoryIndex::transactionFinished(MemoryBackingStoreTransaction& transaction)
{
    auto cleanCursors = std::exchange(m_cleanCursors, { });
    for (WeakPtr cursor : cleanCursors) {
        if (RefPtr protectedCusor = cursor.get()) {
            if (protectedCusor->transaction() != &transaction)
                m_cleanCursors.add(*protectedCusor);
        }
    }

    auto cursors = std::exchange(m_cursors, { });
    for (auto [identifier, cursor] : cursors) {
        if (cursor->transaction() != &transaction)
            m_cursors.add(identifier, cursor);
    }
}

} // namespace IDBServer
} // namespace WebCore
