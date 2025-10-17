/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
#include "IDBStorageRegistry.h"

#include "IDBStorageConnectionToClient.h"
#include <WebCore/UniqueIDBDatabaseConnection.h>
#include <WebCore/UniqueIDBDatabaseTransaction.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

IDBStorageRegistry::IDBStorageRegistry() = default;

IDBStorageRegistry::~IDBStorageRegistry() = default;

WTF_MAKE_TZONE_ALLOCATED_IMPL(IDBStorageRegistry);

WebCore::IDBServer::IDBConnectionToClient& IDBStorageRegistry::ensureConnectionToClient(IPC::Connection::UniqueID connection, WebCore::IDBConnectionIdentifier identifier)
{
    auto addResult = m_connectionsToClient.add(identifier, nullptr);
    if (addResult.isNewEntry)
        addResult.iterator->value = makeUnique<IDBStorageConnectionToClient>(connection, identifier);

    ASSERT(addResult.iterator->value->ipcConnection() == connection);
    return addResult.iterator->value->connectionToClient();
}

void IDBStorageRegistry::removeConnectionToClient(IPC::Connection::UniqueID connection)
{
    auto allConnectionsToClient = std::exchange(m_connectionsToClient, { });
    for (auto& [identifier, connectionToClient] : allConnectionsToClient) {
        if (connectionToClient->ipcConnection() != connection) {
            m_connectionsToClient.add(identifier, WTFMove(connectionToClient));
            continue;
        }
        connectionToClient->connectionToClient().connectionToClientClosed();
    }
}

void IDBStorageRegistry::registerConnection(WebCore::IDBServer::UniqueIDBDatabaseConnection& connection)
{
    auto identifier = connection.identifier();
    ASSERT(!m_connections.contains(identifier));

    m_connections.add(identifier, connection);
}

void IDBStorageRegistry::unregisterConnection(WebCore::IDBServer::UniqueIDBDatabaseConnection& connection)
{
    auto identifier = connection.identifier();
    ASSERT(m_connections.contains(identifier));

    m_connections.remove(identifier);
}

void IDBStorageRegistry::registerTransaction(WebCore::IDBServer::UniqueIDBDatabaseTransaction& transaction)
{
    auto identifier = transaction.info().identifier();
    ASSERT(!m_transactions.contains(identifier));

    m_transactions.add(identifier, transaction);
}

void IDBStorageRegistry::unregisterTransaction(WebCore::IDBServer::UniqueIDBDatabaseTransaction& transaction)
{
    auto identifier = transaction.info().identifier();
    ASSERT(m_transactions.contains(identifier));

    m_transactions.remove(identifier);
}

WebCore::IDBServer::UniqueIDBDatabaseConnection* IDBStorageRegistry::connection(WebCore::IDBDatabaseConnectionIdentifier identifier)
{
    return m_connections.get(identifier).get();
}

WebCore::IDBServer::UniqueIDBDatabaseTransaction* IDBStorageRegistry::transaction(WebCore::IDBResourceIdentifier identifier)
{
    return m_transactions.get(identifier).get();
}

} // namespace WebKit
