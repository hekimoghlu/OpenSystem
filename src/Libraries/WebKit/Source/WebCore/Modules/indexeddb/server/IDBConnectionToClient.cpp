/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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
#include "IDBConnectionToClient.h"

#include "IDBDatabaseNameAndVersion.h"
#include "UniqueIDBDatabaseConnection.h"

namespace WebCore {
namespace IDBServer {

Ref<IDBConnectionToClient> IDBConnectionToClient::create(IDBConnectionToClientDelegate& delegate)
{
    return adoptRef(*new IDBConnectionToClient(delegate));
}

IDBConnectionToClient::IDBConnectionToClient(IDBConnectionToClientDelegate& delegate)
    : m_delegate(&delegate)
{
}

IDBConnectionIdentifier IDBConnectionToClient::identifier() const
{
    ASSERT(m_delegate);
    return *m_delegate->identifier();
}

void IDBConnectionToClient::didDeleteDatabase(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didDeleteDatabase(result);
}

void IDBConnectionToClient::didOpenDatabase(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didOpenDatabase(result);
}

void IDBConnectionToClient::didAbortTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError& error)
{
    if (m_delegate)
        m_delegate->didAbortTransaction(transactionIdentifier, error);
}

void IDBConnectionToClient::didCreateObjectStore(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didCreateObjectStore(result);
}

void IDBConnectionToClient::didDeleteObjectStore(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didDeleteObjectStore(result);
}

void IDBConnectionToClient::didRenameObjectStore(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didRenameObjectStore(result);
}

void IDBConnectionToClient::didClearObjectStore(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didClearObjectStore(result);
}

void IDBConnectionToClient::didCreateIndex(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didCreateIndex(result);
}

void IDBConnectionToClient::didDeleteIndex(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didDeleteIndex(result);
}

void IDBConnectionToClient::didRenameIndex(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didRenameIndex(result);
}

void IDBConnectionToClient::didPutOrAdd(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didPutOrAdd(result);
}

void IDBConnectionToClient::didGetRecord(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didGetRecord(result);
}

void IDBConnectionToClient::didGetAllRecords(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didGetAllRecords(result);
}

void IDBConnectionToClient::didGetCount(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didGetCount(result);
}

void IDBConnectionToClient::didDeleteRecord(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didDeleteRecord(result);
}

void IDBConnectionToClient::didOpenCursor(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didOpenCursor(result);
}

void IDBConnectionToClient::didIterateCursor(const IDBResultData& result)
{
    if (m_delegate)
        m_delegate->didIterateCursor(result);
}

void IDBConnectionToClient::didCommitTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError& error)
{
    if (m_delegate)
        m_delegate->didCommitTransaction(transactionIdentifier, error);
}

void IDBConnectionToClient::fireVersionChangeEvent(UniqueIDBDatabaseConnection& connection, const IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion)
{
    if (m_delegate)
        m_delegate->fireVersionChangeEvent(connection, requestIdentifier, requestedVersion);
}

void IDBConnectionToClient::didStartTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError& error)
{
    if (m_delegate)
        m_delegate->didStartTransaction(transactionIdentifier, error);
}

void IDBConnectionToClient::didCloseFromServer(UniqueIDBDatabaseConnection& connection, const IDBError& error)
{
    if (m_delegate)
        m_delegate->didCloseFromServer(connection, error);
}

void IDBConnectionToClient::notifyOpenDBRequestBlocked(const IDBResourceIdentifier& requestIdentifier, uint64_t oldVersion, uint64_t newVersion)
{
    if (m_delegate)
        m_delegate->notifyOpenDBRequestBlocked(requestIdentifier, oldVersion, newVersion);
}

void IDBConnectionToClient::didGetAllDatabaseNamesAndVersions(const IDBResourceIdentifier& requestIdentifier, Vector<IDBDatabaseNameAndVersion>&& databases)
{
    if (m_delegate)
        m_delegate->didGetAllDatabaseNamesAndVersions(requestIdentifier, WTFMove(databases));
}

void IDBConnectionToClient::registerDatabaseConnection(UniqueIDBDatabaseConnection& connection)
{
    ASSERT(!m_databaseConnections.contains(&connection));
    m_databaseConnections.add(&connection);
}

void IDBConnectionToClient::unregisterDatabaseConnection(UniqueIDBDatabaseConnection& connection)
{
    m_databaseConnections.remove(&connection);
}

void IDBConnectionToClient::connectionToClientClosed()
{
    m_isClosed = true;
    auto databaseConnections = m_databaseConnections;

    for (RefPtr connection : databaseConnections) {
        ASSERT(m_databaseConnections.contains(connection.get()));
        connection->connectionClosedFromClient();
    }

    ASSERT(m_databaseConnections.isEmpty());
}

} // namespace IDBServer
} // namespace WebCore
