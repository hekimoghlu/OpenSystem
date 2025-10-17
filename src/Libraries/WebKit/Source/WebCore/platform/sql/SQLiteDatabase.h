/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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

#include <functional>
#include <sqlite3.h>
#include <wtf/CheckedRef.h>
#include <wtf/Expected.h>
#include <wtf/Lock.h>
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Threading.h>
#include <wtf/UniqueRef.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

#if COMPILER(MSVC)
#pragma warning(disable: 4800)
#endif

struct sqlite3;

namespace WebCore {

class DatabaseAuthorizer;
class SQLiteStatement;
class SQLiteTransaction;

class SQLiteDatabase final : public CanMakeThreadSafeCheckedPtr<SQLiteDatabase> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(SQLiteDatabase, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(SQLiteDatabase);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SQLiteDatabase);

    friend class SQLiteTransaction;
public:
    WEBCORE_EXPORT SQLiteDatabase();
    WEBCORE_EXPORT ~SQLiteDatabase();

    static constexpr ASCIILiteral inMemoryPath() { return ":memory:"_s; }

    enum class OpenMode : uint8_t { ReadOnly, ReadWrite, ReadWriteCreate };
    enum class OpenOptions : uint8_t {
        CanSuspendWhileLocked = 1 << 0,
    };

    WEBCORE_EXPORT bool open(const String& filename, OpenMode = OpenMode::ReadWriteCreate, OptionSet<OpenOptions> = { });
    bool isOpen() const { return m_db; }
    WEBCORE_EXPORT void close();

    WEBCORE_EXPORT int executeSlow(StringView);
    WEBCORE_EXPORT int execute(ASCIILiteral);
    WEBCORE_EXPORT bool executeCommandSlow(StringView);
    WEBCORE_EXPORT bool executeCommand(ASCIILiteral);
    
    WEBCORE_EXPORT bool tableExists(StringView);
    WEBCORE_EXPORT String tableSQL(StringView);
    WEBCORE_EXPORT String indexSQL(StringView);
    WEBCORE_EXPORT void clearAllTables();
    WEBCORE_EXPORT int runVacuumCommand();
    WEBCORE_EXPORT int runIncrementalVacuumCommand();
    
    bool transactionInProgress() const { return m_transactionInProgress; }

    WEBCORE_EXPORT Expected<SQLiteStatement, int> prepareStatementSlow(StringView query);
    WEBCORE_EXPORT Expected<SQLiteStatement, int> prepareStatement(ASCIILiteral query);
    WEBCORE_EXPORT Expected<UniqueRef<SQLiteStatement>, int> prepareHeapStatementSlow(StringView query);
    WEBCORE_EXPORT Expected<UniqueRef<SQLiteStatement>, int> prepareHeapStatement(ASCIILiteral query);

    // Aborts the current database operation. This is thread safe.
    WEBCORE_EXPORT void interrupt();

    int64_t lastInsertRowID();

    // This function returns the number of rows modified, inserted or deleted by the most recently completed INSERT, UPDATE or DELETE statement.
    WEBCORE_EXPORT int lastChanges();

    void setBusyTimeout(int ms);
    void setBusyHandler(int(*)(void*, int));
    
    void setFullsync(bool);
    
    // This enables automatic WAL truncation via a commit hook that uses SQLITE_CHECKPOINT_TRUNCATE.
    // However, it shouldn't be used if you use a custom busy handler or timeout. This is because
    // SQLITE_CHECKPOINT_TRUNCATE will invoke the busy handler if it can't acquire the necessary
    // locks, which can lead to unintended delays.
    void enableAutomaticWALTruncation();
    enum class CheckpointMode : uint8_t { Full, Truncate };
    void checkpoint(CheckpointMode);

    // Gets/sets the maximum size in bytes
    // Depending on per-database attributes, the size will only be settable in units that are the page size of the database, which is established at creation
    // These chunks will never be anything other than 512, 1024, 2048, 4096, 8192, 16384, or 32768 bytes in size.
    // setMaximumSize() will round the size down to the next smallest chunk if the passed size doesn't align.
    int64_t maximumSize();
    WEBCORE_EXPORT void setMaximumSize(int64_t);
    
    // Gets the number of unused bytes in the database file.
    int64_t freeSpaceSize();
    int64_t totalSize();

    // The SQLite SYNCHRONOUS pragma can be either FULL, NORMAL, or OFF
    // FULL - Any writing calls to the DB block until the data is actually on the disk surface
    // NORMAL - SQLite pauses at some critical moments when writing, but much less than FULL
    // OFF - Calls return immediately after the data has been passed to disk
    enum SynchronousPragma { SyncOff = 0, SyncNormal = 1, SyncFull = 2 };
    void setSynchronous(SynchronousPragma);
    
    WEBCORE_EXPORT int lastError();
    WEBCORE_EXPORT const char* lastErrorMsg();
    
    sqlite3* sqlite3Handle() const
    {
#if !PLATFORM(IOS_FAMILY)
        ASSERT(m_sharable || m_openingThread == &Thread::current() || !m_db);
#endif
        return m_db;
    }
    
    void setAuthorizer(DatabaseAuthorizer&);

    Lock& databaseMutex() { return m_lockingMutex; }
    bool isAutoCommitOn() const;

    // The SQLite AUTO_VACUUM pragma can be either NONE, FULL, or INCREMENTAL.
    // NONE - SQLite does not do any vacuuming
    // FULL - SQLite moves all empty pages to the end of the DB file and truncates
    //        the file to remove those pages after every transaction. This option
    //        requires SQLite to store additional information about each page in
    //        the database file.
    // INCREMENTAL - SQLite stores extra information for each page in the database
    //               file, but removes the empty pages only when PRAGMA INCREMANTAL_VACUUM
    //               is called.
    enum AutoVacuumPragma { AutoVacuumNone = 0, AutoVacuumFull = 1, AutoVacuumIncremental = 2 };
    WEBCORE_EXPORT bool turnOnIncrementalAutoVacuum();

    WEBCORE_EXPORT void setCollationFunction(const String& collationName, Function<int(int, const void*, int, const void*)>&&);

    // Set this flag to allow access from multiple threads.  Not all multi-threaded accesses are safe!
    // See http://www.sqlite.org/cvstrac/wiki?p=MultiThreading for more info.
#if ASSERT_ENABLED
    WEBCORE_EXPORT void disableThreadingChecks();
#else
    void disableThreadingChecks() { }
#endif

    WEBCORE_EXPORT static void useFastMalloc();

    WEBCORE_EXPORT static void setIsDatabaseOpeningForbidden(bool);

    WEBCORE_EXPORT void releaseMemory();

    void incrementStatementCount();
    void decrementStatementCount();

private:
    static int authorizerFunction(void*, int, const char*, const char*, const char*, const char*);

    void enableAuthorizer(bool enable) WTF_REQUIRES_LOCK(m_authorizerLock);
    bool useWALJournalMode();

    int pageSize();

    void overrideUnauthorizedFunctions();

    sqlite3* m_db { nullptr };
    int m_pageSize { -1 };
    
    bool m_transactionInProgress { false };
#if ASSERT_ENABLED
    bool m_sharable { false };
    std::atomic<unsigned> m_statementCount { 0 };
#endif

    bool m_useWAL { false };

    Lock m_authorizerLock;
    RefPtr<DatabaseAuthorizer> m_authorizer WTF_GUARDED_BY_LOCK(m_authorizerLock);

    Lock m_lockingMutex;
    RefPtr<Thread> m_openingThread { nullptr };

    Lock m_databaseClosingMutex;

    int m_openError { SQLITE_ERROR };
    CString m_openErrorMessage;
};

inline void SQLiteDatabase::incrementStatementCount()
{
#if ASSERT_ENABLED
    ++m_statementCount;
#endif
}

inline void SQLiteDatabase::decrementStatementCount()
{
#if ASSERT_ENABLED
    ASSERT(m_statementCount);
    --m_statementCount;
#endif
}

} // namespace WebCore
