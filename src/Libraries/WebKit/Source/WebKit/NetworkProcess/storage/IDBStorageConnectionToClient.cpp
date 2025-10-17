/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 4, 2022.
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
#include "IDBStorageConnectionToClient.h"

#include "WebIDBConnectionToServerMessages.h"
#include "WebIDBResult.h"
#include <WebCore/IDBRequestData.h>
#include <WebCore/IDBResultData.h>
#include <WebCore/UniqueIDBDatabaseConnection.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IDBStorageConnectionToClient);

IDBStorageConnectionToClient::IDBStorageConnectionToClient(IPC::Connection::UniqueID connection, WebCore::IDBConnectionIdentifier identifier)
    : m_connection(connection)
    , m_identifier(identifier)
    , m_connectionToClient(WebCore::IDBServer::IDBConnectionToClient::create(*this))
{
}

IDBStorageConnectionToClient::~IDBStorageConnectionToClient()
{
    m_connectionToClient->clearDelegate();
}

WebCore::IDBServer::IDBConnectionToClient& IDBStorageConnectionToClient::connectionToClient()
{
    return m_connectionToClient;
}

void IDBStorageConnectionToClient::didDeleteDatabase(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidDeleteDatabase(resultData), 0);
}

void IDBStorageConnectionToClient::didOpenDatabase(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidOpenDatabase(resultData), 0);
}

void IDBStorageConnectionToClient::didStartTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError& error)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidStartTransaction(transactionIdentifier, error), 0);
}

void IDBStorageConnectionToClient::didAbortTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError& error)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidAbortTransaction(transactionIdentifier, error), 0);
}

void IDBStorageConnectionToClient::didCommitTransaction(const WebCore::IDBResourceIdentifier& transactionIdentifier, const WebCore::IDBError& error)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidCommitTransaction(transactionIdentifier, error), 0);
}

void IDBStorageConnectionToClient::didCreateObjectStore(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidCreateObjectStore(resultData), 0);
}

void IDBStorageConnectionToClient::didDeleteObjectStore(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidDeleteObjectStore(resultData), 0);
}

void IDBStorageConnectionToClient::didRenameObjectStore(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidRenameObjectStore(resultData), 0);
}

void IDBStorageConnectionToClient::didClearObjectStore(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidClearObjectStore(resultData), 0);
}

void IDBStorageConnectionToClient::didCreateIndex(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidCreateIndex(resultData), 0);
}

void IDBStorageConnectionToClient::didDeleteIndex(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidDeleteIndex(resultData), 0);
}

void IDBStorageConnectionToClient::didRenameIndex(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidRenameIndex(resultData), 0);
}

void IDBStorageConnectionToClient::didPutOrAdd(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidPutOrAdd(resultData), 0);
}

void IDBStorageConnectionToClient::didGetRecord(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidGetRecord(resultData), 0);
}

void IDBStorageConnectionToClient::didGetAllRecords(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidGetAllRecords(resultData), 0);
}

void IDBStorageConnectionToClient::didGetCount(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidGetCount(resultData), 0);
}

void IDBStorageConnectionToClient::didDeleteRecord(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidDeleteRecord(resultData), 0);
}

void IDBStorageConnectionToClient::didOpenCursor(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidOpenCursor(resultData), 0);
}

void IDBStorageConnectionToClient::didIterateCursor(const WebCore::IDBResultData& resultData)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidIterateCursor(resultData), 0);
}

void IDBStorageConnectionToClient::didGetAllDatabaseNamesAndVersions(const WebCore::IDBResourceIdentifier& requestIdentifier, Vector<WebCore::IDBDatabaseNameAndVersion>&& databases)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidGetAllDatabaseNamesAndVersions(requestIdentifier, databases), 0);
}

void IDBStorageConnectionToClient::fireVersionChangeEvent(WebCore::IDBServer::UniqueIDBDatabaseConnection& connection, const WebCore::IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::FireVersionChangeEvent(connection.identifier(), requestIdentifier, requestedVersion), 0);
}

void IDBStorageConnectionToClient::didCloseFromServer(WebCore::IDBServer::UniqueIDBDatabaseConnection& connection, const WebCore::IDBError& error)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::DidCloseFromServer(connection.identifier(), error), 0);
}

void IDBStorageConnectionToClient::notifyOpenDBRequestBlocked(const WebCore::IDBResourceIdentifier& requestIdentifier, uint64_t oldVersion, uint64_t newVersion)
{
    IPC::Connection::send(m_connection, Messages::WebIDBConnectionToServer::NotifyOpenDBRequestBlocked(requestIdentifier, oldVersion, newVersion), 0);
}

} // namespace WebKit
