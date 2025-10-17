/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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
#include "IDBTransactionInfo.h"
#include <wtf/Deque.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebCore {

class IDBCursorInfo;
class IDBDatabaseInfo;
class IDBIndexInfo;
class IDBKeyData;
class IDBObjectStoreInfo;
class IDBRequestData;
class IDBValue;

struct IDBGetAllRecordsData;
struct IDBGetRecordData;
struct IDBIterateCursorData;
struct IDBKeyRangeData;

namespace IDBServer {

class IDBServer;
class UniqueIDBDatabase;
class UniqueIDBDatabaseConnection;

class UniqueIDBDatabaseTransaction : public RefCountedAndCanMakeWeakPtr<UniqueIDBDatabaseTransaction> {
public:
    static Ref<UniqueIDBDatabaseTransaction> create(UniqueIDBDatabaseConnection&, const IDBTransactionInfo&);

    WEBCORE_EXPORT ~UniqueIDBDatabaseTransaction();

    UniqueIDBDatabaseConnection* databaseConnection() const;
    UniqueIDBDatabase* database() const;
    const IDBTransactionInfo& info() const { return m_transactionInfo; }
    WEBCORE_EXPORT bool isVersionChange() const;
    bool isReadOnly() const;

    IDBDatabaseInfo* originalDatabaseInfo() const;

    WEBCORE_EXPORT void abort();
    WEBCORE_EXPORT void abortWithoutCallback();
    bool shouldAbortDueToUnhandledRequestError(uint64_t handledRequestResultsCount) const;
    WEBCORE_EXPORT void commit(uint64_t handledRequestResultsCount);

    WEBCORE_EXPORT void createObjectStore(const IDBRequestData&, const IDBObjectStoreInfo&);
    WEBCORE_EXPORT void deleteObjectStore(const IDBRequestData&, const String& objectStoreName);
    WEBCORE_EXPORT void renameObjectStore(const IDBRequestData&, IDBObjectStoreIdentifier, const String& newName);
    WEBCORE_EXPORT void clearObjectStore(const IDBRequestData&, IDBObjectStoreIdentifier);
    WEBCORE_EXPORT void createIndex(const IDBRequestData&, const IDBIndexInfo&);
    WEBCORE_EXPORT void deleteIndex(const IDBRequestData&, IDBObjectStoreIdentifier, const String& indexName);
    WEBCORE_EXPORT void renameIndex(const IDBRequestData&, IDBObjectStoreIdentifier, IDBIndexIdentifier, const String& newName);
    WEBCORE_EXPORT void putOrAdd(const IDBRequestData&, const IDBKeyData&, const IDBValue&, IndexedDB::ObjectStoreOverwriteMode);
    WEBCORE_EXPORT void getRecord(const IDBRequestData&, const IDBGetRecordData&);
    WEBCORE_EXPORT void getAllRecords(const IDBRequestData&, const IDBGetAllRecordsData&);
    WEBCORE_EXPORT void getCount(const IDBRequestData&, const IDBKeyRangeData&);
    WEBCORE_EXPORT void deleteRecord(const IDBRequestData&, const IDBKeyRangeData&);
    WEBCORE_EXPORT void openCursor(const IDBRequestData&, const IDBCursorInfo&);
    WEBCORE_EXPORT void iterateCursor(const IDBRequestData&, const IDBIterateCursorData&);

    void didActivateInBackingStore(const IDBError&);

    const Vector<IDBObjectStoreIdentifier>& objectStoreIdentifiers();

    void setSuspensionAbortResult(const IDBError& error) { m_suspensionAbortResult = { error }; }
    const std::optional<IDBError>& suspensionAbortResult() const { return m_suspensionAbortResult; }

private:
    UniqueIDBDatabaseTransaction(UniqueIDBDatabaseConnection&, const IDBTransactionInfo&);

    WeakPtr<UniqueIDBDatabaseConnection> m_databaseConnection;
    IDBTransactionInfo m_transactionInfo;

    std::unique_ptr<IDBDatabaseInfo> m_originalDatabaseInfo;

    Vector<IDBObjectStoreIdentifier> m_objectStoreIdentifiers;

    std::optional<IDBError> m_suspensionAbortResult;
    Vector<IDBError> m_requestResults;
};

} // namespace IDBServer
} // namespace WebCore
