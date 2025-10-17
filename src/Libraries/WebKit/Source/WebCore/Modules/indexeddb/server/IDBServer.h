/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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

#include "IDBConnectionToClient.h"
#include "IDBDatabaseIdentifier.h"
#include "IDBObjectStoreIdentifier.h"
#include "UniqueIDBDatabase.h"
#include "UniqueIDBDatabaseConnection.h"
#include "UniqueIDBDatabaseManager.h"
#include <pal/HysteresisActivity.h>
#include <pal/SessionID.h>
#include <wtf/CrossThreadTaskHandler.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class IDBCursorInfo;
class IDBRequestData;
class IDBValue;

struct IDBGetRecordData;

namespace IDBServer {

class IDBServer : public UniqueIDBDatabaseManager {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(IDBServer, WEBCORE_EXPORT);
public:
    using SpaceRequester = Function<bool(const ClientOrigin&, uint64_t spaceRequested)>;
    WEBCORE_EXPORT IDBServer(const String& databaseDirectoryPath, SpaceRequester&&, Lock&);
    WEBCORE_EXPORT ~IDBServer();

    WEBCORE_EXPORT void registerConnection(IDBConnectionToClient&);
    WEBCORE_EXPORT void unregisterConnection(IDBConnectionToClient&);

    // Operations requested by the client.
    WEBCORE_EXPORT void openDatabase(const IDBOpenRequestData&);
    WEBCORE_EXPORT void deleteDatabase(const IDBOpenRequestData&);
    WEBCORE_EXPORT void abortTransaction(const IDBResourceIdentifier&);
    WEBCORE_EXPORT void commitTransaction(const IDBResourceIdentifier&, uint64_t handledRequestResultsCount);
    WEBCORE_EXPORT void didFinishHandlingVersionChangeTransaction(IDBDatabaseConnectionIdentifier, const IDBResourceIdentifier&);
    WEBCORE_EXPORT void createObjectStore(const IDBRequestData&, const IDBObjectStoreInfo&);
    WEBCORE_EXPORT void renameObjectStore(const IDBRequestData&, IDBObjectStoreIdentifier, const String& newName);
    WEBCORE_EXPORT void deleteObjectStore(const IDBRequestData&, const String& objectStoreName);
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

    WEBCORE_EXPORT void establishTransaction(IDBDatabaseConnectionIdentifier, const IDBTransactionInfo&);
    WEBCORE_EXPORT void databaseConnectionPendingClose(IDBDatabaseConnectionIdentifier);
    WEBCORE_EXPORT void databaseConnectionClosed(IDBDatabaseConnectionIdentifier);
    WEBCORE_EXPORT void abortOpenAndUpgradeNeeded(IDBDatabaseConnectionIdentifier, const std::optional<IDBResourceIdentifier>& transactionIdentifier);
    WEBCORE_EXPORT void didFireVersionChangeEvent(IDBDatabaseConnectionIdentifier, const IDBResourceIdentifier& requestIdentifier, IndexedDB::ConnectionClosedOnBehalfOfServer);
    WEBCORE_EXPORT void openDBRequestCancelled(const IDBOpenRequestData&);

    WEBCORE_EXPORT void getAllDatabaseNamesAndVersions(IDBConnectionIdentifier, const IDBResourceIdentifier&, const ClientOrigin&);

    // UniqueIDBDatabaseManager
    void registerConnection(UniqueIDBDatabaseConnection&) final;
    void unregisterConnection(UniqueIDBDatabaseConnection&) final;
    void registerTransaction(UniqueIDBDatabaseTransaction&) final;
    void unregisterTransaction(UniqueIDBDatabaseTransaction&) final;
    std::unique_ptr<IDBBackingStore> createBackingStore(const IDBDatabaseIdentifier&) final;
    void requestSpace(const ClientOrigin&, uint64_t size, CompletionHandler<void(bool)>&&) final;

    WEBCORE_EXPORT HashSet<SecurityOriginData> getOrigins() const;
    WEBCORE_EXPORT void closeAndDeleteDatabasesModifiedSince(WallTime);
    WEBCORE_EXPORT void closeAndDeleteDatabasesForOrigins(const Vector<SecurityOriginData>&);
    void closeDatabasesForOrigins(const Vector<SecurityOriginData>&, Function<bool(const SecurityOriginData&, const ClientOrigin&)>&&);
    WEBCORE_EXPORT void renameOrigin(const WebCore::SecurityOriginData&, const WebCore::SecurityOriginData&);

    WEBCORE_EXPORT static uint64_t diskUsage(const String& rootDirectory, const ClientOrigin&);

private:
    UniqueIDBDatabase& getOrCreateUniqueIDBDatabase(const IDBDatabaseIdentifier&);
    UniqueIDBDatabaseTransaction* idbTransaction(const IDBRequestData&) const;

    void upgradeFilesIfNecessary();
    String upgradedDatabaseDirectory(const WebCore::IDBDatabaseIdentifier&);
    void removeDatabasesModifiedSinceForVersion(WallTime, const String&);
    void removeDatabasesWithOriginsForVersion(const Vector<SecurityOriginData>&, const String&);

    HashMap<IDBConnectionIdentifier, RefPtr<IDBConnectionToClient>> m_connectionMap;
    HashMap<IDBDatabaseIdentifier, std::unique_ptr<UniqueIDBDatabase>> m_uniqueIDBDatabaseMap;

    HashMap<IDBDatabaseConnectionIdentifier, UniqueIDBDatabaseConnection*> m_databaseConnections;
    HashMap<IDBResourceIdentifier, UniqueIDBDatabaseTransaction*> m_transactions;

    HashMap<uint64_t, Function<void ()>> m_deleteDatabaseCompletionHandlers;

    String m_databaseDirectoryPath;

    SpaceRequester m_spaceRequester;

    Lock& m_lock;
};

} // namespace IDBServer
} // namespace WebCore
