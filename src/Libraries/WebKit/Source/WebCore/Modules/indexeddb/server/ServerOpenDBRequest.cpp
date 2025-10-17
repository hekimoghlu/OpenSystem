/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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
#include "ServerOpenDBRequest.h"

#include "IDBResultData.h"

namespace WebCore {
namespace IDBServer {

Ref<ServerOpenDBRequest> ServerOpenDBRequest::create(IDBConnectionToClient& connection, const IDBOpenRequestData& requestData)
{
    return adoptRef(*new ServerOpenDBRequest(connection, requestData));
}

ServerOpenDBRequest::ServerOpenDBRequest(IDBConnectionToClient& connection, const IDBOpenRequestData& requestData)
    : m_connection(connection)
    , m_requestData(requestData)
{
}

bool ServerOpenDBRequest::isOpenRequest() const
{
    return m_requestData.isOpenRequest();
}

bool ServerOpenDBRequest::isDeleteRequest() const
{
    return m_requestData.isDeleteRequest();
}

void ServerOpenDBRequest::maybeNotifyRequestBlocked(uint64_t currentVersion)
{
    if (m_notifiedBlocked)
        return;

    uint64_t requestedVersion = isOpenRequest() ?  m_requestData.requestedVersion() : 0;
    m_connection->notifyOpenDBRequestBlocked(m_requestData.requestIdentifier(), currentVersion, requestedVersion);

    m_notifiedBlocked = true;
}

void ServerOpenDBRequest::notifyDidDeleteDatabase(const IDBDatabaseInfo& info)
{
    ASSERT(isDeleteRequest());

    m_connection->didDeleteDatabase(IDBResultData::deleteDatabaseSuccess(m_requestData.requestIdentifier(), info));
}

void ServerOpenDBRequest::notifiedConnectionsOfVersionChange(HashSet<IDBDatabaseConnectionIdentifier>&& connectionIdentifiers)
{
    ASSERT(!m_notifiedConnectionsOfVersionChange);

    m_notifiedConnectionsOfVersionChange = true;
    m_connectionsPendingVersionChangeEvent = WTFMove(connectionIdentifiers);
}

void ServerOpenDBRequest::connectionClosedOrFiredVersionChangeEvent(IDBDatabaseConnectionIdentifier connectionIdentifier)
{
    m_connectionsPendingVersionChangeEvent.remove(connectionIdentifier);
}

} // namespace IDBServer
} // namespace WebCore
