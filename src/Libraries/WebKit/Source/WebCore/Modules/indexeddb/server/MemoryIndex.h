/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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

#include "IDBIndexInfo.h"
#include "IDBResourceIdentifier.h"
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {

class IDBCursorInfo;
class IDBError;
class IDBGetAllResult;
class IDBGetResult;
class IDBKeyData;
class IndexKey;
class ThreadSafeDataBuffer;

struct IDBKeyRangeData;

namespace IndexedDB {
enum class GetAllType : bool;
enum class IndexRecordType : bool;
}

namespace IDBServer {

class IndexValueStore;
class MemoryBackingStoreTransaction;
class MemoryIndexCursor;
class MemoryObjectStore;

class MemoryIndex : public RefCounted<MemoryIndex>, public CanMakeThreadSafeCheckedPtr<MemoryIndex> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MemoryIndex);
public:
    static Ref<MemoryIndex> create(const IDBIndexInfo&, MemoryObjectStore&);

    ~MemoryIndex();

    const IDBIndexInfo& info() const { return m_info; }

    void rename(const String& newName) { m_info.rename(newName); }

    IDBGetResult getResultForKeyRange(IndexedDB::IndexRecordType, const IDBKeyRangeData&) const;
    uint64_t countForKeyRange(const IDBKeyRangeData&);
    void getAllRecords(const IDBKeyRangeData&, std::optional<uint32_t> count, IndexedDB::GetAllType, IDBGetAllResult&) const;

    IDBError putIndexKey(const IDBKeyData&, const IndexKey&);

    void removeEntriesWithValueKey(const IDBKeyData&);
    void removeRecord(const IDBKeyData&, const IndexKey&);

    void objectStoreCleared();
    void clearIndexValueStore();
    void replaceIndexValueStore(std::unique_ptr<IndexValueStore>&&);

    MemoryIndexCursor* maybeOpenCursor(const IDBCursorInfo&, MemoryBackingStoreTransaction&);
    IndexValueStore* valueStore() { return m_records.get(); }

    WeakPtr<MemoryObjectStore> objectStore();
    RefPtr<MemoryObjectStore> protectedObjectStore();

    void cursorDidBecomeClean(MemoryIndexCursor&);
    void cursorDidBecomeDirty(MemoryIndexCursor&);

    void notifyCursorsOfValueChange(const IDBKeyData& indexKey, const IDBKeyData& primaryKey);
    void transactionFinished(MemoryBackingStoreTransaction&);

private:
    MemoryIndex(const IDBIndexInfo&, MemoryObjectStore&);

    uint64_t recordCountForKey(const IDBKeyData&) const;

    void notifyCursorsOfAllRecordsChanged();

    IDBIndexInfo m_info;
    WeakPtr<MemoryObjectStore> m_objectStore;

    std::unique_ptr<IndexValueStore> m_records;

    HashMap<IDBResourceIdentifier, RefPtr<MemoryIndexCursor>> m_cursors;
    WeakHashSet<MemoryIndexCursor> m_cleanCursors;
};

} // namespace IDBServer
} // namespace WebCore
