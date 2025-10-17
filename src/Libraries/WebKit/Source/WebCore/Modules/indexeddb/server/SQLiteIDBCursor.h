/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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

#include "IDBCursorRecord.h"
#include "IDBIndexIdentifier.h"
#include "IDBIndexInfo.h"
#include "IDBKeyData.h"
#include "IDBKeyRangeData.h"
#include "IDBObjectStoreIdentifier.h"
#include "IDBResourceIdentifier.h"
#include "IDBValue.h"
#include "SQLiteStatement.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Deque.h>
#include <wtf/Markable.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class IDBCursorInfo;
class IDBGetResult;

namespace IDBServer {

class SQLiteIDBTransaction;

class SQLiteIDBCursor {
    WTF_MAKE_TZONE_ALLOCATED(SQLiteIDBCursor);
    WTF_MAKE_NONCOPYABLE(SQLiteIDBCursor);
public:
    static std::unique_ptr<SQLiteIDBCursor> maybeCreate(SQLiteIDBTransaction&, const IDBCursorInfo&);
    static std::unique_ptr<SQLiteIDBCursor> maybeCreateBackingStoreCursor(SQLiteIDBTransaction&, IDBObjectStoreIdentifier, std::optional<IDBIndexIdentifier>, const IDBKeyRangeData&);

    SQLiteIDBCursor(SQLiteIDBTransaction&, const IDBCursorInfo&);
    SQLiteIDBCursor(SQLiteIDBTransaction&, IDBObjectStoreIdentifier, std::optional<IDBIndexIdentifier>, const IDBKeyRangeData&);

    ~SQLiteIDBCursor();

    const IDBResourceIdentifier& identifier() const { return m_cursorIdentifier; }
    SQLiteIDBTransaction* transaction() const;

    IDBObjectStoreIdentifier objectStoreID() const { return m_objectStoreID; }
    int64_t currentRecordRowID() const;

    const IDBKeyData& currentKey() const;
    const IDBKeyData& currentPrimaryKey() const;
    const IDBValue& currentValue() const;

    bool advance(uint64_t count);
    bool iterate(const IDBKeyData& targetKey, const IDBKeyData& targetPrimaryKey);
    bool prefetchOneRecord();
    bool prefetch();

    bool didComplete() const;
    bool didError() const;

    void objectStoreRecordsChanged();

    enum class ShouldIncludePrefetchedRecords : bool { No, Yes };
    void currentData(IDBGetResult&, const std::optional<IDBKeyPath>&, ShouldIncludePrefetchedRecords = ShouldIncludePrefetchedRecords::No);

private:
    bool establishStatement();
    bool createSQLiteStatement(StringView sql);
    bool bindArguments();

    bool resetAndRebindPreIndexStatementIfNecessary();
    void resetAndRebindStatement();

    enum class FetchResult {
        Success,
        Failure,
        ShouldFetchAgain
    };

    bool fetch();

    struct SQLiteCursorRecord {
        IDBCursorRecord record;
        bool completed { false };
        bool errored { false };
        int64_t rowID { 0 };
        bool isTerminalRecord() const { return completed || errored; }
    };
    bool fetchNextRecord(SQLiteCursorRecord&);
    FetchResult internalFetchNextRecord(SQLiteCursorRecord&);

    void markAsErrored(SQLiteCursorRecord&);

    bool isDirectionNext() const { return m_cursorDirection == IndexedDB::CursorDirection::Next || m_cursorDirection == IndexedDB::CursorDirection::Nextunique; }

    void increaseCountToPrefetch();

    uint64_t boundIDValue() const;

    CheckedPtr<SQLiteIDBTransaction> m_transaction;
    IDBResourceIdentifier m_cursorIdentifier;
    IDBObjectStoreIdentifier m_objectStoreID;
    Markable<IDBIndexIdentifier> m_indexID;
    IndexedDB::CursorDirection m_cursorDirection { IndexedDB::CursorDirection::Next };
    IndexedDB::CursorType m_cursorType;
    IDBKeyRangeData m_keyRange;

    IDBKeyData m_currentLowerKey;
    IDBKeyData m_currentUpperKey;
    IDBKeyData m_currentIndexRecordValue;

    Deque<SQLiteCursorRecord> m_fetchedRecords;
    uint64_t m_fetchedRecordsSize { 0 };
    IDBKeyData m_currentKeyForUniqueness;

    std::unique_ptr<SQLiteStatement> m_preIndexStatement;
    std::unique_ptr<SQLiteStatement> m_statement;
    std::unique_ptr<SQLiteStatement> m_cachedObjectStoreStatement;

    bool m_statementNeedsReset { true };
    std::variant<IDBObjectStoreIdentifier, IDBIndexIdentifier> m_boundID;

    bool m_backingStoreCursor { false };

    unsigned m_prefetchCount { 0 };
};

} // namespace IDBServer
} // namespace WebCore
