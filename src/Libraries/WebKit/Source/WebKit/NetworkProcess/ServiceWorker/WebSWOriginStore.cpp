/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
#include "WebSWOriginStore.h"

#include "MessageSenderInlines.h"
#include "WebSWClientConnectionMessages.h"
#include "WebSWServerConnection.h"
#include <WebCore/SecurityOrigin.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSWOriginStore);

using namespace WebCore;

WebSWOriginStore::WebSWOriginStore()
    : m_store(*this)
{
}

void WebSWOriginStore::addToStore(const SecurityOriginData& origin)
{
    m_store.scheduleAddition(computeSharedStringHash(origin.toString()));
    m_store.flushPendingChanges();
}

void WebSWOriginStore::removeFromStore(const SecurityOriginData& origin)
{
    m_store.scheduleRemoval(computeSharedStringHash(origin.toString()));
    m_store.flushPendingChanges();
}

void WebSWOriginStore::clearStore()
{
    m_store.clear();
}

void WebSWOriginStore::importComplete()
{
    m_isImported = true;
    for (Ref connection : m_webSWServerConnections)
        connection->send(Messages::WebSWClientConnection::SetSWOriginTableIsImported());
}

void WebSWOriginStore::registerSWServerConnection(WebSWServerConnection& connection)
{
    m_webSWServerConnections.add(connection);

    if (!m_store.isEmpty())
        sendStoreHandle(connection);

    if (m_isImported)
        connection.send(Messages::WebSWClientConnection::SetSWOriginTableIsImported());
}

void WebSWOriginStore::unregisterSWServerConnection(WebSWServerConnection& connection)
{
    m_webSWServerConnections.remove(connection);
}

void WebSWOriginStore::sendStoreHandle(WebSWServerConnection& connection)
{
    auto handle = m_store.createSharedMemoryHandle();
    if (!handle)
        return;
    connection.send(Messages::WebSWClientConnection::SetSWOriginTableSharedMemory(WTFMove(*handle)));
}

void WebSWOriginStore::didInvalidateSharedMemory()
{
    for (Ref connection : m_webSWServerConnections)
        sendStoreHandle(connection.get());
}

} // namespace WebKit
