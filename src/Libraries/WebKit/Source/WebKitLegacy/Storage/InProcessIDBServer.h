/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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

#include <WebCore/IDBConnectionToClient.h>
#include <WebCore/IDBConnectionToServer.h>
#include <WebCore/IDBIndexIdentifier.h>
#include <WebCore/IDBObjectStoreIdentifier.h>
#include <WebCore/IDBServer.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WTF {
class WorkQueue;
}

namespace PAL {
class SessionID;
}

namespace WebCore {
struct ClientOrigin;

namespace IDBClient {
class IDBConnectionToServer;
}

namespace IDBServer {
class IDBServer;
}
} // namespace WebCore

class InProcessIDBServer final : public WebCore::IDBClient::IDBConnectionToServerDelegate, public WebCore::IDBServer::IDBConnectionToClientDelegate, public ThreadSafeRefCounted<InProcessIDBServer> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(InProcessIDBServer);
public:
    static Ref<InProcessIDBServer> create(PAL::SessionID);
    static Ref<InProcessIDBServer> create(PAL::SessionID, const String& databaseDirectoryPath);

    virtual ~InProcessIDBServer();

    WebCore::IDBClient::IDBConnectionToServer& connectionToServer() const;
    WebCore::IDBServer::IDBConnectionToClient& connectionToClient() const;
    WebCore::IDBServer::IDBServer& server() { return *m_server; }

    // IDBConnectionToServer
    void deleteDatabase(const WebCore::IDBOpenRequestData&) final;
    void openDatabase(const WebCore::IDBOpenRequestData&) final;
    void abortTransaction(const WebCore::IDBResourceIdentifier&) final;
    void commitTransaction(const WebCore::IDBResourceIdentifier&, uint64_t pendingCountRequest) final;
    void didFinishHandlingVersionChangeTransaction(WebCore::IDBDatabaseConnectionIdentifier, const WebCore::IDBResourceIdentifier&) final;
    void createObjectStore(const WebCore::IDBRequestData&, const WebCore::IDBObjectStoreInfo&) final;
    void deleteObjectStore(const WebCore::IDBRequestData&, const String& objectStoreName) final;
    void renameObjectStore(const WebCore::IDBRequestData&, WebCore::IDBObjectStoreIdentifier, const String& newName) final;
    void clearObjectStore(const WebCore::IDBRequestData&, WebCore::IDBObjectStoreIdentifier) final;
    void createIndex(const WebCore::IDBRequestData&, const WebCore::IDBIndexInfo&) final;
    void deleteIndex(const WebCore::IDBRequestData&, WebCore::IDBObjectStoreIdentifier, const String& indexName) final;
    void renameIndex(const WebCore::IDBRequestData&, WebCore::IDBObjectStoreIdentifier, WebCore::IDBIndexIdentifier, const String& newName) final;
    void putOrAdd(const WebCore::IDBRequestData&, const WebCore::IDBKeyData&, const WebCore::IDBValue&, const WebCore::IndexedDB::ObjectStoreOverwriteMode) final;
    void getRecord(const WebCore::IDBRequestData&, const WebCore::IDBGetRecordData&) final;
    void getAllRecords(const WebCore::IDBRequestData&, const WebCore::IDBGetAllRecordsData&) final;
    void getCount(const WebCore::IDBRequestData&, const WebCore::IDBKeyRangeData&) final;
    void deleteRecord(const WebCore::IDBRequestData&, const WebCore::IDBKeyRangeData&) final;
    void openCursor(const WebCore::IDBRequestData&, const WebCore::IDBCursorInfo&) final;
    void iterateCursor(const WebCore::IDBRequestData&, const WebCore::IDBIterateCursorData&) final;
    void establishTransaction(WebCore::IDBDatabaseConnectionIdentifier, const WebCore::IDBTransactionInfo&) final;
    void databaseConnectionPendingClose(WebCore::IDBDatabaseConnectionIdentifier) final;
    void databaseConnectionClosed(WebCore::IDBDatabaseConnectionIdentifier) final;
    void abortOpenAndUpgradeNeeded(WebCore::IDBDatabaseConnectionIdentifier, const std::optional<WebCore::IDBResourceIdentifier>& transactionIdentifier) final;
    void didFireVersionChangeEvent(WebCore::IDBDatabaseConnectionIdentifier, const WebCore::IDBResourceIdentifier& requestIdentifier, const WebCore::IndexedDB::ConnectionClosedOnBehalfOfServer) final;
    void openDBRequestCancelled(const WebCore::IDBOpenRequestData&) final;
    void getAllDatabaseNamesAndVersions(const WebCore::IDBResourceIdentifier&, const WebCore::ClientOrigin&) final;

    // IDBConnectionToClient
    std::optional<WebCore::IDBConnectionIdentifier> identifier() const final;
    void didDeleteDatabase(const WebCore::IDBResultData&) final;
    void didOpenDatabase(const WebCore::IDBResultData&) final;
    void didAbortTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError&) final;
    void didCommitTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError&) final;
    void didCreateObjectStore(const WebCore::IDBResultData&) final;
    void didDeleteObjectStore(const WebCore::IDBResultData&) final;
    void didRenameObjectStore(const WebCore::IDBResultData&) final;
    void didClearObjectStore(const WebCore::IDBResultData&) final;
    void didCreateIndex(const WebCore::IDBResultData&) final;
    void didDeleteIndex(const WebCore::IDBResultData&) final;
    void didRenameIndex(const WebCore::IDBResultData&) final;
    void didPutOrAdd(const WebCore::IDBResultData&) final;
    void didGetRecord(const WebCore::IDBResultData&) final;
    void didGetAllRecords(const WebCore::IDBResultData&) final;
    void didGetCount(const WebCore::IDBResultData&) final;
    void didDeleteRecord(const WebCore::IDBResultData&) final;
    void didOpenCursor(const WebCore::IDBResultData&) final;
    void didIterateCursor(const WebCore::IDBResultData&) final;
    void fireVersionChangeEvent(WebCore::IDBServer::UniqueIDBDatabaseConnection&, const WebCore::IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion) final;
    void didStartTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError&) final;
    void didCloseFromServer(WebCore::IDBServer::UniqueIDBDatabaseConnection&, const WebCore::IDBError&) final;
    void notifyOpenDBRequestBlocked(const WebCore::IDBResourceIdentifier& requestIdentifier, uint64_t oldVersion, uint64_t newVersion) final;
    void didGetAllDatabaseNamesAndVersions(const WebCore::IDBResourceIdentifier&, Vector<WebCore::IDBDatabaseNameAndVersion>&&) final;

    void closeAndDeleteDatabasesModifiedSince(WallTime);

    void dispatchTask(Function<void()>&&);
    void dispatchTaskReply(Function<void()>&&);

private:
    InProcessIDBServer(PAL::SessionID, const String& databaseDirectoryPath = nullString());

    Lock m_serverLock;
    std::unique_ptr<WebCore::IDBServer::IDBServer> m_server;
    RefPtr<WebCore::IDBClient::IDBConnectionToServer> m_connectionToServer;
    RefPtr<WebCore::IDBServer::IDBConnectionToClient> m_connectionToClient;
    Ref<WTF::WorkQueue> m_queue;
};
