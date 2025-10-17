/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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
#include "SQLiteIDBTransaction.h"

#include "IDBCursorInfo.h"
#include "IndexedDB.h"
#include "Logging.h"
#include "SQLiteIDBBackingStore.h"
#include "SQLiteIDBCursor.h"
#include "SQLiteTransaction.h"
#include <wtf/FileSystem.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace IDBServer {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SQLiteIDBTransaction);

SQLiteIDBTransaction::SQLiteIDBTransaction(SQLiteIDBBackingStore& backingStore, const IDBTransactionInfo& info)
    : m_info(info)
    , m_backingStore(backingStore)
{
}

SQLiteIDBTransaction::~SQLiteIDBTransaction()
{
    if (inProgress())
        m_sqliteTransaction->rollback();

    // Explicitly clear cursors, as that also unregisters them from the backing store.
    clearCursors();
}

IDBError SQLiteIDBTransaction::begin(SQLiteDatabase& database)
{
    ASSERT(!m_sqliteTransaction);

    if (isReadOnly()) {
        m_sqliteDatabase = &database;
        return IDBError { };
    }

    m_sqliteTransaction = makeUnique<SQLiteTransaction>(database, true);
    m_sqliteTransaction->begin();

    if (m_sqliteTransaction->inProgress())
        return IDBError { };

    return IDBError { ExceptionCode::UnknownError, "Could not start SQLite transaction in database backend"_s };
}

IDBError SQLiteIDBTransaction::commit()
{
    LOG(IndexedDB, "SQLiteIDBTransaction::commit");

    if (isReadOnly()) {
        reset();
        return IDBError { };
    }

    if (!m_sqliteTransaction || !m_sqliteTransaction->inProgress())
        return IDBError { ExceptionCode::UnknownError, "No SQLite transaction in progress to commit"_s };

    m_sqliteTransaction->commit();
    if (m_sqliteTransaction->inProgress())
        return IDBError { ExceptionCode::UnknownError, "Unable to commit SQLite transaction in database backend"_s };

    deleteBlobFilesIfNecessary();
    moveBlobFilesIfNecessary();

    reset();
    return IDBError { };
}

void SQLiteIDBTransaction::moveBlobFilesIfNecessary()
{
    ASSERT(!isReadOnly());

    String databaseDirectory = m_backingStore->databaseDirectory();
    for (auto& entry : m_blobTemporaryAndStoredFilenames) {
        if (!FileSystem::hardLinkOrCopyFile(entry.first, FileSystem::pathByAppendingComponent(databaseDirectory, entry.second)))
            LOG_ERROR("Failed to link/copy temporary blob file '%s' to location '%s'", entry.first.utf8().data(), FileSystem::pathByAppendingComponent(databaseDirectory, entry.second).utf8().data());

        FileSystem::deleteFile(entry.first);
    }

    m_blobTemporaryAndStoredFilenames.clear();
}

void SQLiteIDBTransaction::deleteBlobFilesIfNecessary()
{
    ASSERT(!isReadOnly());

    if (m_blobRemovedFilenames.isEmpty())
        return;

    String databaseDirectory = m_backingStore->databaseDirectory();
    for (auto& entry : m_blobRemovedFilenames) {
        String fullPath = FileSystem::pathByAppendingComponent(databaseDirectory, entry);

        FileSystem::deleteFile(fullPath);
    }

    m_blobRemovedFilenames.clear();
}

IDBError SQLiteIDBTransaction::abort()
{
    if (isReadOnly()) {
        reset();
        return IDBError { };
    }

    for (auto& entry : m_blobTemporaryAndStoredFilenames)
        FileSystem::deleteFile(entry.first);

    m_blobTemporaryAndStoredFilenames.clear();

    if (!m_sqliteTransaction || !m_sqliteTransaction->inProgress())
        return IDBError { ExceptionCode::UnknownError, "No SQLite transaction in progress to abort"_s };

    m_sqliteTransaction->rollback();

    if (m_sqliteTransaction->inProgress())
        return IDBError { ExceptionCode::UnknownError, "Unable to abort SQLite transaction in database backend"_s };

    reset();
    return IDBError { };
}

void SQLiteIDBTransaction::reset()
{
    m_sqliteTransaction = nullptr;
    clearCursors();
    ASSERT(m_blobTemporaryAndStoredFilenames.isEmpty());
}

std::unique_ptr<SQLiteIDBCursor> SQLiteIDBTransaction::maybeOpenBackingStoreCursor(IDBObjectStoreIdentifier objectStoreID, std::optional<IDBIndexIdentifier> indexID, const IDBKeyRangeData& range)
{
    ASSERT(inProgressOrReadOnly());

    auto cursor = SQLiteIDBCursor::maybeCreateBackingStoreCursor(*this, objectStoreID, indexID, range);

    if (cursor)
        m_backingStoreCursors.add(cursor.get());

    return cursor;
}

SQLiteIDBCursor* SQLiteIDBTransaction::maybeOpenCursor(const IDBCursorInfo& info)
{
    if (m_sqliteTransaction && !m_sqliteTransaction->inProgress())
        return nullptr;

    auto addResult = m_cursors.add(info.identifier(), SQLiteIDBCursor::maybeCreate(*this, info));
    ASSERT(addResult.isNewEntry);

    // It is possible the cursor failed to create and we just stored a null value.
    if (!addResult.iterator->value) {
        m_cursors.remove(addResult.iterator);
        return nullptr;
    }

    return addResult.iterator->value.get();
}

void SQLiteIDBTransaction::closeCursor(SQLiteIDBCursor& cursor)
{
    auto backingStoreTake = m_backingStoreCursors.take(&cursor);
    if (backingStoreTake) {
        ASSERT(!m_cursors.contains(cursor.identifier()));
        return;
    }

    ASSERT(m_cursors.contains(cursor.identifier()));

    m_backingStore->unregisterCursor(cursor);
    m_cursors.remove(cursor.identifier());
}

void SQLiteIDBTransaction::notifyCursorsOfChanges(IDBObjectStoreIdentifier objectStoreID)
{
    ASSERT(!isReadOnly());

    for (auto& i : m_cursors) {
        if (i.value->objectStoreID() == objectStoreID)
            i.value->objectStoreRecordsChanged();
    }

    for (auto* cursor : m_backingStoreCursors) {
        if (cursor->objectStoreID() == objectStoreID)
            cursor->objectStoreRecordsChanged();
    }
}

void SQLiteIDBTransaction::clearCursors()
{
    for (auto& cursor : m_cursors.values())
        m_backingStore->unregisterCursor(*cursor);

    m_cursors.clear();
}

bool SQLiteIDBTransaction::inProgress() const
{
    return m_sqliteTransaction && m_sqliteTransaction->inProgress();
}

bool SQLiteIDBTransaction::inProgressOrReadOnly() const
{
    return isReadOnly() || inProgress();
}

void SQLiteIDBTransaction::addBlobFile(const String& temporaryPath, const String& storedFilename)
{
    ASSERT(!isReadOnly());

    m_blobTemporaryAndStoredFilenames.append({ temporaryPath, storedFilename });
}

void SQLiteIDBTransaction::addRemovedBlobFile(const String& removedFilename)
{
    ASSERT(!isReadOnly());
    ASSERT(!m_blobRemovedFilenames.contains(removedFilename));

    m_blobRemovedFilenames.add(removedFilename);
}

SQLiteDatabase* SQLiteIDBTransaction::sqliteDatabase() const
{
    if (m_sqliteTransaction)
        return &m_sqliteTransaction->database();

    return m_sqliteDatabase.get();
}


} // namespace IDBServer
} // namespace WebCore
