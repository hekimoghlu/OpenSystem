/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
#include "IndexValueStore.h"

#include "IDBError.h"
#include "IDBKeyRangeData.h"
#include "Logging.h"
#include "MemoryIndex.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {
namespace IDBServer {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IndexValueStore);

IndexValueStore::IndexValueStore(bool unique)
    : m_unique(unique)
{
}

const IDBKeyData* IndexValueStore::lowestValueForKey(const IDBKeyData& key) const
{
    const auto& entry = m_records.get(key);
    if (!entry)
        return nullptr;

    return entry->getLowest();
}

Vector<IDBKeyData> IndexValueStore::allValuesForKey(const IDBKeyData& key, uint32_t limit) const
{
    const auto& entry = m_records.get(key);
    if (!entry)
        return { };

    Vector<IDBKeyData> results;
    for (auto iterator = entry->begin(); results.size() < limit && iterator.isValid(); ++iterator)
        results.append(iterator.key());

    return results;
}

uint64_t IndexValueStore::countForKey(const IDBKeyData& key) const
{
    const auto& entry = m_records.get(key);
    if (!entry)
        return 0;

    return entry->getCount();
}

bool IndexValueStore::contains(const IDBKeyData& key) const
{
    const auto& entry = m_records.get(key);
    if (!entry)
        return false;

    ASSERT(entry->getCount());

    return true;
}

IDBError IndexValueStore::addRecord(const IDBKeyData& indexKey, const IDBKeyData& valueKey)
{
    auto result = m_records.add(indexKey, nullptr);

    if (!result.isNewEntry && m_unique)
        return IDBError(ExceptionCode::ConstraintError);

    if (result.isNewEntry)
        result.iterator->value = makeUnique<IndexValueEntry>(m_unique);

    result.iterator->value->addKey(valueKey);
    m_orderedKeys.insert(indexKey);

    return IDBError { };
}

void IndexValueStore::removeRecord(const IDBKeyData& indexKey, const IDBKeyData& valueKey)
{
    auto iterator = m_records.find(indexKey);
    if (iterator == m_records.end())
        return;

    if (!iterator->value)
        return;

    if (iterator->value->removeKey(valueKey) && !iterator->value->getCount()) {
        m_records.remove(iterator);
        m_orderedKeys.erase(indexKey);
    }
}

void IndexValueStore::removeEntriesWithValueKey(MemoryIndex& index, const IDBKeyData& valueKey)
{
    Vector<IDBKeyData> entryKeysToRemove;
    entryKeysToRemove.reserveInitialCapacity(m_records.size());

    for (auto& entry : m_records) {
        if (entry.value->removeKey(valueKey))
            index.notifyCursorsOfValueChange(entry.key, valueKey);
        if (!entry.value->getCount())
            entryKeysToRemove.append(entry.key);
    }

    for (auto& entry : entryKeysToRemove) {
        m_orderedKeys.erase(entry);
        m_records.remove(entry);
    }
}

IDBKeyData IndexValueStore::lowestKeyWithRecordInRange(const IDBKeyRangeData& range) const
{
    LOG(IndexedDB, "IndexValueStore::lowestKeyWithRecordInRange - %s", range.loggingString().utf8().data());

    if (range.isExactlyOneKey())
        return m_records.contains(range.lowerKey) ? range.lowerKey : IDBKeyData();

    auto iterator = lowestIteratorInRange(range);
    if (iterator == m_orderedKeys.end())
        return { };

    return *iterator;
}

IDBKeyDataSet::iterator IndexValueStore::lowestIteratorInRange(const IDBKeyRangeData& range) const
{
    auto lowestInRange = m_orderedKeys.lower_bound(range.lowerKey);

    if (lowestInRange == m_orderedKeys.end())
        return lowestInRange;

    if (range.lowerOpen && *lowestInRange == range.lowerKey) {
        ++lowestInRange;

        if (lowestInRange == m_orderedKeys.end())
            return lowestInRange;
    }

    if (!range.upperKey.isNull()) {
        if (lowestInRange->compare(range.upperKey) > 0)
            return m_orderedKeys.end();
        if (range.upperOpen && *lowestInRange == range.upperKey)
            return m_orderedKeys.end();
    }

    return lowestInRange;
}

IDBKeyDataSet::reverse_iterator IndexValueStore::highestReverseIteratorInRange(const IDBKeyRangeData& range) const
{
    auto highestInRange = IDBKeyDataSet::reverse_iterator(m_orderedKeys.upper_bound(range.upperKey));

    if (highestInRange == m_orderedKeys.rend())
        return highestInRange;

    if (range.upperOpen && *highestInRange == range.upperKey) {
        ++highestInRange;

        if (highestInRange == m_orderedKeys.rend())
            return highestInRange;
    }

    if (!range.lowerKey.isNull()) {
        if (highestInRange->compare(range.lowerKey) < 0)
            return m_orderedKeys.rend();
        if (range.lowerOpen && *highestInRange == range.lowerKey)
            return m_orderedKeys.rend();
    }

    return highestInRange;
}

IndexValueStore::Iterator IndexValueStore::find(const IDBKeyData& key, bool open)
{
    IDBKeyRangeData range;
    if (!key.isNull())
        range.lowerKey = key;
    else
        range.lowerKey = IDBKeyData::minimum();
    range.lowerOpen = open;

    auto iterator = lowestIteratorInRange(range);
    if (iterator == m_orderedKeys.end())
        return { };

    auto record = m_records.get(*iterator);
    ASSERT(record);

    auto primaryIterator = record->begin();
    ASSERT(primaryIterator.isValid());
    return { *this, iterator, primaryIterator };
}

IndexValueStore::Iterator IndexValueStore::find(const IDBKeyData& key, const IDBKeyData& primaryKey)
{
    ASSERT(!key.isNull());
    ASSERT(!primaryKey.isNull());

    IDBKeyRangeData range;
    range.lowerKey = key;
    range.lowerOpen = false;

    auto iterator = lowestIteratorInRange(range);
    if (iterator == m_orderedKeys.end())
        return { };

    auto record = m_records.get(*iterator);
    ASSERT(record);

    // If the main record iterator is not equal to the key we were looking for,
    // we know the primary key record should be the first.
    if (*iterator != key) {
        auto primaryIterator = record->begin();
        ASSERT(primaryIterator.isValid());

        return { *this, iterator, primaryIterator };
    }

    auto primaryIterator = record->find(primaryKey);
    if (primaryIterator.isValid())
        return { *this, iterator, primaryIterator };

    // If we didn't find a primary key iterator in this entry,
    // we need to move on to start of the next record.
    iterator++;
    if (iterator == m_orderedKeys.end())
        return { };

    record = m_records.get(*iterator);
    ASSERT(record);

    primaryIterator = record->begin();
    ASSERT(primaryIterator.isValid());

    return { *this, iterator, primaryIterator };
}

IndexValueStore::Iterator IndexValueStore::reverseFind(const IDBKeyData& key, CursorDuplicity duplicity, bool open)
{
    IDBKeyRangeData range;
    if (!key.isNull())
        range.upperKey = key;
    else
        range.upperKey = IDBKeyData::maximum();
    range.upperOpen = open;

    auto iterator = highestReverseIteratorInRange(range);
    if (iterator == m_orderedKeys.rend())
        return { };

    auto record = m_records.get(*iterator);
    ASSERT(record);

    auto primaryIterator = record->reverseBegin(duplicity);
    ASSERT(primaryIterator.isValid());
    return { *this, duplicity, iterator, primaryIterator };
}

IndexValueStore::Iterator IndexValueStore::reverseFind(const IDBKeyData& key, const IDBKeyData& primaryKey, CursorDuplicity duplicity)
{
    ASSERT(!key.isNull());
    ASSERT(!primaryKey.isNull());

    IDBKeyRangeData range;
    range.upperKey = key;
    range.upperOpen = false;

    auto iterator = highestReverseIteratorInRange(range);
    if (iterator == m_orderedKeys.rend())
        return { };

    auto record = m_records.get(*iterator);
    ASSERT(record);

    auto primaryIterator = record->reverseFind(primaryKey, duplicity);
    if (primaryIterator.isValid())
        return { *this, duplicity, iterator, primaryIterator };

    // If we didn't find a primary key iterator in this entry,
    // we need to move on to start of the next record.
    iterator++;
    if (iterator == m_orderedKeys.rend())
        return { };

    record = m_records.get(*iterator);
    ASSERT(record);

    primaryIterator = record->reverseBegin(duplicity);
    ASSERT(primaryIterator.isValid());

    return { *this, duplicity, iterator, primaryIterator };
}


IndexValueStore::Iterator::Iterator(IndexValueStore& store, IDBKeyDataSet::iterator iterator, IndexValueEntry::Iterator primaryIterator)
    : m_store(&store)
    , m_forwardIterator(iterator)
    , m_primaryKeyIterator(primaryIterator)
{
}

IndexValueStore::Iterator::Iterator(IndexValueStore& store, CursorDuplicity duplicity, IDBKeyDataSet::reverse_iterator iterator, IndexValueEntry::Iterator primaryIterator)
    : m_store(&store)
    , m_forward(false)
    , m_duplicity(duplicity)
    , m_reverseIterator(iterator)
    , m_primaryKeyIterator(primaryIterator)
{
}

IndexValueStore::Iterator& IndexValueStore::Iterator::nextIndexEntry()
{
    if (!m_store)
        return *this;

    if (m_forward) {
        ++m_forwardIterator;
        if (m_forwardIterator == m_store->m_orderedKeys.end()) {
            invalidate();
            return *this;
        }

        auto* entry = m_store->m_records.get(*m_forwardIterator);
        ASSERT(entry);

        m_primaryKeyIterator = entry->begin();
        ASSERT(m_primaryKeyIterator.isValid());
    } else {
        ++m_reverseIterator;
        if (m_reverseIterator == m_store->m_orderedKeys.rend()) {
            invalidate();
            return *this;
        }

        auto* entry = m_store->m_records.get(*m_reverseIterator);
        ASSERT(entry);

        m_primaryKeyIterator = entry->reverseBegin(m_duplicity);
        ASSERT(m_primaryKeyIterator.isValid());
    }
    
    return *this;
}

IndexValueStore::Iterator& IndexValueStore::Iterator::operator++()
{
    if (!isValid())
        return *this;

    ++m_primaryKeyIterator;
    if (m_primaryKeyIterator.isValid())
        return *this;

    // Ran out of primary key records, so move the main index iterator.
    return nextIndexEntry();
}

void IndexValueStore::Iterator::invalidate()
{
    m_store = nullptr;
    m_primaryKeyIterator.invalidate();
}

bool IndexValueStore::Iterator::isValid()
{
    return m_store && m_primaryKeyIterator.isValid();
}

const IDBKeyData& IndexValueStore::Iterator::key()
{
    ASSERT(isValid());
    return m_forward ? *m_forwardIterator : *m_reverseIterator;
}

const IDBKeyData& IndexValueStore::Iterator::primaryKey()
{
    ASSERT(isValid());
    return m_primaryKeyIterator.key();
}

#if !LOG_DISABLED
String IndexValueStore::loggingString() const
{
    StringBuilder builder;
    for (auto& key : m_orderedKeys)
        builder.append("Key: "_s, key.loggingString(), "  Entry has "_s, m_records.get(key)->getCount(), " entries"_s);
    return builder.toString();
}
#endif

} // namespace IDBServer
} // namespace WebCore
