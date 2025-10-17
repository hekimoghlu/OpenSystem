/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 22, 2021.
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

#include "Connection.h"
#include <WebCore/IDBConnectionToClient.h>
#include <WebCore/IDBConnectionToClientDelegate.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class IDBStorageConnectionToClient final : public WebCore::IDBServer::IDBConnectionToClientDelegate {
    WTF_MAKE_TZONE_ALLOCATED(IDBStorageConnectionToClient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(IDBStorageConnectionToClient);
public:
    IDBStorageConnectionToClient(IPC::Connection::UniqueID, WebCore::IDBConnectionIdentifier);
    ~IDBStorageConnectionToClient();

    std::optional<WebCore::IDBConnectionIdentifier> identifier() const final { return m_identifier; }
    IPC::Connection::UniqueID ipcConnection() const { return m_connection; }
    WebCore::IDBServer::IDBConnectionToClient& connectionToClient();

private:
    // IDBConnectionToClientDelegate
    void didDeleteDatabase(const WebCore::IDBResultData&) final;
    void didOpenDatabase(const WebCore::IDBResultData&) final;
    void didAbortTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError&) final;
    void didCommitTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError&) final;
    void didStartTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError&) final;
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
    void didGetAllDatabaseNamesAndVersions(const WebCore::IDBResourceIdentifier&, Vector<WebCore::IDBDatabaseNameAndVersion>&&) final;
    void notifyOpenDBRequestBlocked(const WebCore::IDBResourceIdentifier& requestIdentifier, uint64_t oldVersion, uint64_t newVersion) final;
    void fireVersionChangeEvent(WebCore::IDBServer::UniqueIDBDatabaseConnection&, const WebCore::IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion) final;
    void didCloseFromServer(WebCore::IDBServer::UniqueIDBDatabaseConnection&, const WebCore::IDBError&) final;

    IPC::Connection::UniqueID m_connection;
    WebCore::IDBConnectionIdentifier m_identifier;
    Ref<WebCore::IDBServer::IDBConnectionToClient> m_connectionToClient;
};

} // namespace WebKit
