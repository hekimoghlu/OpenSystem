/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

#include "MessageSender.h"
#include "SandboxExtension.h"
#include <WebCore/IDBConnectionToServer.h>
#include <WebCore/IDBIndexIdentifier.h>
#include <WebCore/IDBObjectStoreIdentifier.h>
#include <WebCore/ProcessIdentifier.h>
#include <optional>

namespace WebKit {

class WebIDBResult;

class WebIDBConnectionToServer final : private WebCore::IDBClient::IDBConnectionToServerDelegate, private IPC::MessageSender, public RefCounted<WebIDBConnectionToServer> {
public:
    static Ref<WebIDBConnectionToServer> create(PAL::SessionID);
    virtual ~WebIDBConnectionToServer();

    WebCore::IDBClient::IDBConnectionToServer& coreConnectionToServer();
    std::optional<WebCore::IDBConnectionIdentifier> identifier() const final;

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);
    void connectionToServerLost();

private:
    WebIDBConnectionToServer(PAL::SessionID);

    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final { return 0; }

    // IDBConnectionToServerDelegate
    void deleteDatabase(const WebCore::IDBOpenRequestData&) final;
    void openDatabase(const WebCore::IDBOpenRequestData&) final;
    void abortTransaction(const WebCore::IDBResourceIdentifier&) final;
    void commitTransaction(const WebCore::IDBResourceIdentifier&, uint64_t handledRequestResultsCount) final;
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

    // Messages received from Network Process
    void didDeleteDatabase(const WebCore::IDBResultData&);
    void didOpenDatabase(const WebCore::IDBResultData&);
    void didAbortTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError&);
    void didCommitTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError&);
    void didCreateObjectStore(const WebCore::IDBResultData&);
    void didDeleteObjectStore(const WebCore::IDBResultData&);
    void didRenameObjectStore(const WebCore::IDBResultData&);
    void didClearObjectStore(const WebCore::IDBResultData&);
    void didCreateIndex(const WebCore::IDBResultData&);
    void didDeleteIndex(const WebCore::IDBResultData&);
    void didRenameIndex(const WebCore::IDBResultData&);
    void didPutOrAdd(const WebCore::IDBResultData&);
    void didGetRecord(const WebIDBResult&);
    void didGetAllRecords(const WebIDBResult&);
    void didGetCount(const WebCore::IDBResultData&);
    void didDeleteRecord(const WebCore::IDBResultData&);
    void didOpenCursor(const WebIDBResult&);
    void didIterateCursor(const WebIDBResult&);
    void fireVersionChangeEvent(WebCore::IDBDatabaseConnectionIdentifier uniqueDatabaseConnectionIdentifier, const WebCore::IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion);
    void didStartTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError&);
    void didCloseFromServer(WebCore::IDBDatabaseConnectionIdentifier, const WebCore::IDBError&);
    void notifyOpenDBRequestBlocked(const WebCore::IDBResourceIdentifier& requestIdentifier, uint64_t oldVersion, uint64_t newVersion);
    void didGetAllDatabaseNamesAndVersions(const WebCore::IDBResourceIdentifier&, Vector<WebCore::IDBDatabaseNameAndVersion>&&);

    Ref<WebCore::IDBClient::IDBConnectionToServer> m_connectionToServer;
};

} // namespace WebKit
