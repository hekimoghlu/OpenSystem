/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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

#include "IDBError.h"
#include "IDBIndexIdentifier.h"
#include "IDBObjectStoreIdentifier.h"
#include "IDBResourceIdentifier.h"
#include "IDBTransactionInfo.h"
#include "IndexedDB.h"
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/RobinHoodHashSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class IDBCursorInfo;
class SQLiteDatabase;
class SQLiteTransaction;
struct IDBKeyRangeData;

namespace IDBServer {

class SQLiteIDBBackingStore;
class SQLiteIDBCursor;

class SQLiteIDBTransaction : public CanMakeThreadSafeCheckedPtr<SQLiteIDBTransaction> {
    WTF_MAKE_TZONE_ALLOCATED(SQLiteIDBTransaction);
    WTF_MAKE_NONCOPYABLE(SQLiteIDBTransaction);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SQLiteIDBTransaction);
public:
    SQLiteIDBTransaction(SQLiteIDBBackingStore&, const IDBTransactionInfo&);
    ~SQLiteIDBTransaction();

    const IDBResourceIdentifier& transactionIdentifier() const { return m_info.identifier(); }

    IDBError begin(SQLiteDatabase&);
    IDBError commit();
    IDBError abort();

    std::unique_ptr<SQLiteIDBCursor> maybeOpenBackingStoreCursor(IDBObjectStoreIdentifier, std::optional<IDBIndexIdentifier>, const IDBKeyRangeData&);
    SQLiteIDBCursor* maybeOpenCursor(const IDBCursorInfo&);

    void closeCursor(SQLiteIDBCursor&);
    void notifyCursorsOfChanges(IDBObjectStoreIdentifier);

    IDBTransactionMode mode() const { return m_info.mode(); }
    IDBTransactionDurability durability() const { return m_info.durability(); }
    bool inProgress() const;
    bool inProgressOrReadOnly() const;

    SQLiteDatabase* sqliteDatabase() const;
    SQLiteTransaction* sqliteTransaction() const { return m_sqliteTransaction.get(); }
    SQLiteIDBBackingStore& backingStore() { return m_backingStore.get(); }

    void addBlobFile(const String& temporaryPath, const String& storedFilename);
    void addRemovedBlobFile(const String& removedFilename);

private:
    void clearCursors();
    void reset();

    void moveBlobFilesIfNecessary();
    void deleteBlobFilesIfNecessary();
    bool isReadOnly() const { return mode() == IDBTransactionMode::Readonly; }

    IDBTransactionInfo m_info;

    CheckedRef<SQLiteIDBBackingStore> m_backingStore;
    CheckedPtr<SQLiteDatabase> m_sqliteDatabase;
    std::unique_ptr<SQLiteTransaction> m_sqliteTransaction;
    HashMap<IDBResourceIdentifier, std::unique_ptr<SQLiteIDBCursor>> m_cursors;
    HashSet<SQLiteIDBCursor*> m_backingStoreCursors;
    Vector<std::pair<String, String>> m_blobTemporaryAndStoredFilenames;
    MemoryCompactRobinHoodHashSet<String> m_blobRemovedFilenames;
};

} // namespace IDBServer
} // namespace WebCore
