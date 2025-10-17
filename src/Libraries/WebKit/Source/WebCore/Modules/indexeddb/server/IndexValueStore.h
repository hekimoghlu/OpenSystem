/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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

#include "IDBCursorInfo.h"
#include "IDBKeyData.h"
#include "IndexValueEntry.h"
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class IDBError;

struct IDBKeyRangeData;

namespace IDBServer {

class MemoryIndex;

typedef HashMap<IDBKeyData, std::unique_ptr<IndexValueEntry>, IDBKeyDataHash, IDBKeyDataHashTraits> IndexKeyValueMap;

class IndexValueStore final : public CanMakeThreadSafeCheckedPtr<IndexValueStore> {
    WTF_MAKE_TZONE_ALLOCATED(IndexValueStore);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(IndexValueStore);
public:
    explicit IndexValueStore(bool unique);

    const IDBKeyData* lowestValueForKey(const IDBKeyData&) const;
    Vector<IDBKeyData> allValuesForKey(const IDBKeyData&, uint32_t limit) const;
    uint64_t countForKey(const IDBKeyData&) const;
    IDBKeyData lowestKeyWithRecordInRange(const IDBKeyRangeData&) const;
    bool contains(const IDBKeyData&) const;

    IDBError addRecord(const IDBKeyData& indexKey, const IDBKeyData& valueKey);
    void removeRecord(const IDBKeyData& indexKey, const IDBKeyData& valueKey);

    void removeEntriesWithValueKey(MemoryIndex&, const IDBKeyData& valueKey);

    class Iterator {
        friend class IndexValueStore;
    public:
        Iterator()
        {
        }

        Iterator(IndexValueStore&, IDBKeyDataSet::iterator, IndexValueEntry::Iterator);
        Iterator(IndexValueStore&, CursorDuplicity, IDBKeyDataSet::reverse_iterator, IndexValueEntry::Iterator);

        void invalidate();
        bool isValid();

        const IDBKeyData& key();
        const IDBKeyData& primaryKey();
        const ThreadSafeDataBuffer& value();

        Iterator& operator++();
        Iterator& nextIndexEntry();

    private:
        CheckedPtr<IndexValueStore> m_store;
        bool m_forward { true };
        CursorDuplicity m_duplicity { CursorDuplicity::Duplicates };
        IDBKeyDataSet::iterator m_forwardIterator;
        IDBKeyDataSet::reverse_iterator m_reverseIterator;

        IndexValueEntry::Iterator m_primaryKeyIterator;
    };

    // Returns an iterator pointing to the first primaryKey record in the requested key, or the next key if it doesn't exist.
    Iterator find(const IDBKeyData&, bool open = false);
    Iterator reverseFind(const IDBKeyData&, CursorDuplicity, bool open = false);

    // Returns an iterator pointing to the key/primaryKey record, or the next one after it if it doesn't exist.
    Iterator find(const IDBKeyData&, const IDBKeyData& primaryKey);
    Iterator reverseFind(const IDBKeyData&, const IDBKeyData& primaryKey, CursorDuplicity);

#if !LOG_DISABLED
    String loggingString() const;
#endif

private:
    IDBKeyDataSet::iterator lowestIteratorInRange(const IDBKeyRangeData&) const;
    IDBKeyDataSet::reverse_iterator highestReverseIteratorInRange(const IDBKeyRangeData&) const;

    IndexKeyValueMap m_records;
    IDBKeyDataSet m_orderedKeys;
    
    bool m_unique;
};

} // namespace IDBServer
} // namespace WebCore
