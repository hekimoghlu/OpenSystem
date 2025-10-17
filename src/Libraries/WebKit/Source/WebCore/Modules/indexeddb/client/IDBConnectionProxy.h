/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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

#include "IDBConnectionToServer.h"
#include "IDBDatabaseNameAndVersionRequest.h"
#include "IDBIndexIdentifier.h"
#include "IDBObjectStoreIdentifier.h"
#include "IDBResourceIdentifier.h"
#include "TransactionOperation.h"
#include <wtf/CrossThreadQueue.h>
#include <wtf/CrossThreadTask.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/MainThread.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class IDBDatabase;
class IDBDatabaseIdentifier;
class IDBError;
class IDBOpenDBRequest;
class IDBOpenRequestData;
class IDBResultData;
class IDBTransaction;
class ScriptExecutionContext;
class SecurityOrigin;

struct IDBGetRecordData;
struct IDBIterateCursorData;

namespace IDBClient {

class IDBConnectionToServer;

class WEBCORE_EXPORT IDBConnectionProxy final {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(IDBConnectionProxy, WEBCORE_EXPORT);
public:
    IDBConnectionProxy(IDBConnectionToServer&, PAL::SessionID);

    Ref<IDBOpenDBRequest> openDatabase(ScriptExecutionContext&, const IDBDatabaseIdentifier&, uint64_t version);
    void didOpenDatabase(const IDBResultData&);

    Ref<IDBOpenDBRequest> deleteDatabase(ScriptExecutionContext&, const IDBDatabaseIdentifier&);
    void didDeleteDatabase(const IDBResultData&);

    void createObjectStore(TransactionOperation&, const IDBObjectStoreInfo&);
    void deleteObjectStore(TransactionOperation&, const String& objectStoreName);
    void clearObjectStore(TransactionOperation&, IDBObjectStoreIdentifier);
    void createIndex(TransactionOperation&, const IDBIndexInfo&);
    void deleteIndex(TransactionOperation&, IDBObjectStoreIdentifier, const String& indexName);
    void putOrAdd(TransactionOperation&, IDBKeyData&&, const IDBValue&, const IndexedDB::ObjectStoreOverwriteMode);
    void getRecord(TransactionOperation&, const IDBGetRecordData&);
    void getAllRecords(TransactionOperation&, const IDBGetAllRecordsData&);
    void getCount(TransactionOperation&, const IDBKeyRangeData&);
    void deleteRecord(TransactionOperation&, const IDBKeyRangeData&);
    void openCursor(TransactionOperation&, const IDBCursorInfo&);
    void iterateCursor(TransactionOperation&, const IDBIterateCursorData&);
    void renameObjectStore(TransactionOperation&, IDBObjectStoreIdentifier, const String& newName);
    void renameIndex(TransactionOperation&, IDBObjectStoreIdentifier, IDBIndexIdentifier, const String& newName);

    void fireVersionChangeEvent(IDBDatabaseConnectionIdentifier, const IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion);
    void didFireVersionChangeEvent(IDBDatabaseConnectionIdentifier, const IDBResourceIdentifier& requestIdentifier, const IndexedDB::ConnectionClosedOnBehalfOfServer = IndexedDB::ConnectionClosedOnBehalfOfServer::No);

    void notifyOpenDBRequestBlocked(const IDBResourceIdentifier& requestIdentifier, uint64_t oldVersion, uint64_t newVersion);
    void openDBRequestCancelled(const IDBOpenRequestData&);

    void establishTransaction(IDBTransaction&);
    void commitTransaction(IDBTransaction&, uint64_t handledRequestResultsCount);
    void abortTransaction(IDBTransaction&);

    void didStartTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&);
    void didCommitTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&);
    void didAbortTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&);

    void didFinishHandlingVersionChangeTransaction(IDBDatabaseConnectionIdentifier, IDBTransaction&);
    void databaseConnectionPendingClose(IDBDatabase&);
    void databaseConnectionClosed(IDBDatabase&);

    void didCloseFromServer(IDBDatabaseConnectionIdentifier, const IDBError&);

    void connectionToServerLost(const IDBError&);

    void abortOpenAndUpgradeNeeded(IDBDatabaseConnectionIdentifier, const std::optional<IDBResourceIdentifier>& transactionIdentifier);

    void completeOperation(const IDBResultData&);

    IDBConnectionIdentifier serverConnectionIdentifier() const { return m_serverConnectionIdentifier; }

    void ref();
    void deref();

    void getAllDatabaseNamesAndVersions(ScriptExecutionContext&, Function<void(std::optional<Vector<IDBDatabaseNameAndVersion>>&&)>&&);
    void didGetAllDatabaseNamesAndVersions(const IDBResourceIdentifier&, std::optional<Vector<IDBDatabaseNameAndVersion>>&&);

    void registerDatabaseConnection(IDBDatabase&, ScriptExecutionContextIdentifier);
    void unregisterDatabaseConnection(IDBDatabase&);

    void forgetActiveOperations(const Vector<RefPtr<TransactionOperation>>&);
    void forgetTransaction(IDBTransaction&);
    void abortActivitiesForCurrentThread();
    void setContextSuspended(ScriptExecutionContext& currentContext, bool isContextSuspended);

    PAL::SessionID sessionID() const;

private:
    void completeOpenDBRequest(const IDBResultData&);
    bool hasRecordOfTransaction(const IDBTransaction&) const WTF_REQUIRES_LOCK(m_transactionMapLock);

    void saveOperation(TransactionOperation&);
    std::pair<RefPtr<IDBDatabase>, std::optional<ScriptExecutionContextIdentifier>> databaseFromConnectionIdentifier(IDBDatabaseConnectionIdentifier);

    template<typename... Parameters, typename... Arguments>
    void callConnectionOnMainThread(void (IDBConnectionToServer::*method)(Parameters...), Arguments&&... arguments)
    {
        if (isMainThread())
            (m_connectionToServer.get().*method)(std::forward<Arguments>(arguments)...);
        else
            postMainThreadTask(m_connectionToServer.get(), method, arguments...);
    }

    template<typename... Arguments>
    void postMainThreadTask(Arguments&&... arguments)
    {
        auto task = createCrossThreadTask(arguments...);
        m_mainThreadQueue.append(WTFMove(task));

        scheduleMainThreadTasks();
    }

    void scheduleMainThreadTasks();
    void handleMainThreadTasks();

    CheckedRef<IDBConnectionToServer> m_connectionToServer;
    IDBConnectionIdentifier m_serverConnectionIdentifier;

    Lock m_databaseConnectionMapLock;
    Lock m_openDBRequestMapLock;
    Lock m_transactionMapLock;
    Lock m_transactionOperationLock;
    Lock m_databaseInfoMapLock;
    Lock m_mainThreadTaskLock;

    struct WeakIDBDatabase {
        ThreadSafeWeakPtr<IDBDatabase> database;
        std::optional<ScriptExecutionContextIdentifier> contextIdentifier;
    };
    HashMap<IDBDatabaseConnectionIdentifier, WeakIDBDatabase> m_databaseConnectionMap WTF_GUARDED_BY_LOCK(m_databaseConnectionMapLock);
    HashMap<IDBResourceIdentifier, RefPtr<IDBOpenDBRequest>> m_openDBRequestMap WTF_GUARDED_BY_LOCK(m_openDBRequestMapLock);
    HashMap<IDBResourceIdentifier, RefPtr<IDBTransaction>> m_pendingTransactions WTF_GUARDED_BY_LOCK(m_transactionMapLock);
    HashMap<IDBResourceIdentifier, RefPtr<IDBTransaction>> m_committingTransactions WTF_GUARDED_BY_LOCK(m_transactionMapLock);
    HashMap<IDBResourceIdentifier, RefPtr<IDBTransaction>> m_abortingTransactions WTF_GUARDED_BY_LOCK(m_transactionMapLock);
    HashMap<IDBResourceIdentifier, RefPtr<TransactionOperation>> m_activeOperations WTF_GUARDED_BY_LOCK(m_transactionOperationLock);
    HashMap<IDBResourceIdentifier, Ref<IDBDatabaseNameAndVersionRequest>> m_databaseInfoCallbacks WTF_GUARDED_BY_LOCK(m_databaseInfoMapLock);

    CrossThreadQueue<CrossThreadTask> m_mainThreadQueue;
    RefPtr<IDBConnectionToServer> m_mainThreadProtector WTF_GUARDED_BY_LOCK(m_mainThreadTaskLock);
    PAL::SessionID m_sessionID;
};

} // namespace IDBClient
} // namespace WebCore
