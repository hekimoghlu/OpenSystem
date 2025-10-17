/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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

#include "IDBDatabaseInfo.h"
#include "IDBKeyData.h"
#include "IDBTransactionInfo.h"
#include "IndexValueStore.h"
#include "ThreadSafeDataBuffer.h"
#include <wtf/CheckedRef.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace IDBServer {

class MemoryCursor;
class MemoryIDBBackingStore;
class MemoryIndex;
class MemoryObjectStore;

typedef HashMap<IDBKeyData, ThreadSafeDataBuffer, IDBKeyDataHash, IDBKeyDataHashTraits> KeyValueMap;

class MemoryBackingStoreTransaction : public RefCountedAndCanMakeWeakPtr<MemoryBackingStoreTransaction> {
public:
    static Ref<MemoryBackingStoreTransaction> create(MemoryIDBBackingStore&, const IDBTransactionInfo&);

    MemoryBackingStoreTransaction(MemoryIDBBackingStore&, const IDBTransactionInfo&);
    ~MemoryBackingStoreTransaction();

    bool isVersionChange() const { return m_info.mode() == IDBTransactionMode::Versionchange; }
    bool isWriting() const { return m_info.mode() != IDBTransactionMode::Readonly; }
    bool isAborting() const { return m_isAborting; }

    const IDBDatabaseInfo& originalDatabaseInfo() const;

    void addNewObjectStore(MemoryObjectStore&);
    void addExistingObjectStore(MemoryObjectStore&);
    
    void recordValueChanged(MemoryObjectStore&, const IDBKeyData&, ThreadSafeDataBuffer*);
    void objectStoreDeleted(Ref<MemoryObjectStore>&&);
    void objectStoreCleared(MemoryObjectStore&, std::unique_ptr<KeyValueMap>&&, std::unique_ptr<IDBKeyDataSet>&&);
    void objectStoreRenamed(MemoryObjectStore&, const String& oldName);
    void indexRenamed(MemoryIndex&, const String& oldName);
    void indexCleared(MemoryIndex&, std::unique_ptr<IndexValueStore>&&);

    void addNewIndex(MemoryIndex&);
    void addExistingIndex(MemoryIndex&);
    void indexDeleted(Ref<MemoryIndex>&&);

    void abort();
    void commit();

    IDBTransactionInfo info() const { return m_info; }
    MemoryCursor* cursor(const IDBResourceIdentifier&) const;
    void addCursor(MemoryCursor&);

private:
    void finish();

    CheckedRef<MemoryIDBBackingStore> m_backingStore;
    IDBTransactionInfo m_info;

    std::unique_ptr<IDBDatabaseInfo> m_originalDatabaseInfo;

    bool m_inProgress { true };
    bool m_isAborting { false };

    HashSet<RefPtr<MemoryObjectStore>> m_objectStores;
    HashSet<RefPtr<MemoryObjectStore>> m_versionChangeAddedObjectStores;
    HashSet<RefPtr<MemoryIndex>> m_indexes;
    HashSet<RefPtr<MemoryIndex>> m_versionChangeAddedIndexes;

    HashMap<MemoryObjectStore*, uint64_t> m_originalKeyGenerators;
    HashMap<String, RefPtr<MemoryObjectStore>> m_deletedObjectStores;
    HashSet<RefPtr<MemoryIndex>> m_deletedIndexes;
    HashMap<MemoryObjectStore*, std::unique_ptr<KeyValueMap>> m_originalValues;
    HashMap<MemoryObjectStore*, std::unique_ptr<KeyValueMap>> m_clearedKeyValueMaps;
    HashMap<MemoryObjectStore*, std::unique_ptr<IDBKeyDataSet>> m_clearedOrderedKeys;
    HashMap<MemoryObjectStore*, String> m_originalObjectStoreNames;
    HashMap<MemoryIndex*, String> m_originalIndexNames;
    HashMap<MemoryIndex*, std::unique_ptr<IndexValueStore>> m_clearedIndexValueStores;

    HashMap<IDBResourceIdentifier, WeakPtr<MemoryCursor>> m_cursors;
};

} // namespace IDBServer
} // namespace WebCore
