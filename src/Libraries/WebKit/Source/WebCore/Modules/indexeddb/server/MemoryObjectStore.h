/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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

#include "IDBIndexIdentifier.h"
#include "IDBKeyData.h"
#include "IDBObjectStoreInfo.h"
#include "IndexKey.h"
#include "MemoryIndex.h"
#include "MemoryObjectStoreCursor.h"
#include "ThreadSafeDataBuffer.h"
#include <wtf/HashMap.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebCore {

class IDBCursorInfo;
class IDBError;
class IDBGetAllResult;
class IDBKeyData;
class IDBValue;

struct IDBKeyRangeData;

namespace IndexedDB {
enum class GetAllType : bool;
enum class IndexRecordType : bool;
}

namespace IDBServer {

class MemoryBackingStoreTransaction;

typedef HashMap<IDBKeyData, ThreadSafeDataBuffer, IDBKeyDataHash, IDBKeyDataHashTraits> KeyValueMap;

class MemoryObjectStore : public RefCountedAndCanMakeWeakPtr<MemoryObjectStore> {
public:
    static Ref<MemoryObjectStore> create(const IDBObjectStoreInfo&);

    ~MemoryObjectStore();

    void transactionFinished(MemoryBackingStoreTransaction&);
    void writeTransactionStarted(MemoryBackingStoreTransaction&);
    void writeTransactionFinished(MemoryBackingStoreTransaction&);

    MemoryBackingStoreTransaction* writeTransaction();

    IDBError createIndex(MemoryBackingStoreTransaction&, const IDBIndexInfo&);
    IDBError deleteIndex(MemoryBackingStoreTransaction&, IDBIndexIdentifier);
    void deleteAllIndexes(MemoryBackingStoreTransaction&);
    void registerIndex(Ref<MemoryIndex>&&);

    bool containsRecord(const IDBKeyData&);
    void deleteRecord(const IDBKeyData&);
    void deleteRange(const IDBKeyRangeData&);
    IDBError addRecord(MemoryBackingStoreTransaction&, const IDBKeyData&, const IDBValue&);
    IDBError addRecord(MemoryBackingStoreTransaction&, const IDBKeyData&, const IndexIDToIndexKeyMap&, const IDBValue&);

    uint64_t currentKeyGeneratorValue() const { return m_keyGeneratorValue; }
    void setKeyGeneratorValue(uint64_t value) { m_keyGeneratorValue = value; }

    void clear();
    void replaceKeyValueStore(std::unique_ptr<KeyValueMap>&&, std::unique_ptr<IDBKeyDataSet>&&);

    ThreadSafeDataBuffer valueForKey(const IDBKeyData&) const;
    ThreadSafeDataBuffer valueForKeyRange(const IDBKeyRangeData&) const;
    IDBKeyData lowestKeyWithRecordInRange(const IDBKeyRangeData&) const;
    IDBGetResult indexValueForKeyRange(IDBIndexIdentifier, IndexedDB::IndexRecordType, const IDBKeyRangeData&) const;
    uint64_t countForKeyRange(std::optional<IDBIndexIdentifier>, const IDBKeyRangeData&) const;

    void getAllRecords(const IDBKeyRangeData&, std::optional<uint32_t> count, IndexedDB::GetAllType, IDBGetAllResult&) const;

    const IDBObjectStoreInfo& info() const { return m_info; }
    IDBObjectStoreInfo& info() { return m_info; }

    MemoryObjectStoreCursor* maybeOpenCursor(const IDBCursorInfo&, MemoryBackingStoreTransaction&);

    IDBKeyDataSet* orderedKeys() { return m_orderedKeys.get(); }

    MemoryIndex* indexForIdentifier(IDBIndexIdentifier);

    void maybeRestoreDeletedIndex(Ref<MemoryIndex>&&);

    void rename(const String& newName) { m_info.rename(newName); }
    void renameIndex(MemoryIndex&, const String& newName);

    RefPtr<MemoryIndex> takeIndexByIdentifier(IDBIndexIdentifier);

private:
    MemoryObjectStore(const IDBObjectStoreInfo&);

    IDBKeyDataSet::iterator lowestIteratorInRange(const IDBKeyRangeData&, bool reverse) const;

    IDBError populateIndexWithExistingRecords(MemoryIndex&);
    IDBError updateIndexesForPutRecord(const IDBKeyData&, const IndexIDToIndexKeyMap&);
    void updateIndexesForDeleteRecord(const IDBKeyData& value);
    void updateCursorsForPutRecord(IDBKeyDataSet::iterator);
    void updateCursorsForDeleteRecord(const IDBKeyData&);

    IDBObjectStoreInfo m_info;

    WeakPtr<MemoryBackingStoreTransaction> m_writeTransaction;
    uint64_t m_keyGeneratorValue { 1 };

    std::unique_ptr<KeyValueMap> m_keyValueStore;
    std::unique_ptr<IDBKeyDataSet> m_orderedKeys;

    void unregisterIndex(MemoryIndex&);
    HashMap<IDBIndexIdentifier, RefPtr<MemoryIndex>> m_indexesByIdentifier;
    HashMap<String, RefPtr<MemoryIndex>> m_indexesByName;
    HashMap<IDBResourceIdentifier, RefPtr<MemoryObjectStoreCursor>> m_cursors;
};

} // namespace IDBServer
} // namespace WebCore
