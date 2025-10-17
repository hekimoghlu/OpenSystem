/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#include <WebCore/IDBDatabaseConnectionIdentifier.h>
#include <WebCore/IDBResourceIdentifier.h>
#include <wtf/CheckedPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
namespace IDBServer {
class UniqueIDBDatabaseConnection;
class UniqueIDBDatabaseTransaction;
}
}

namespace WebKit {

class IDBStorageConnectionToClient;

class IDBStorageRegistry : public CanMakeThreadSafeCheckedPtr<IDBStorageRegistry> {
    WTF_MAKE_TZONE_ALLOCATED(IDBStorageRegistry);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(IDBStorageRegistry);
public:
    IDBStorageRegistry();
    ~IDBStorageRegistry();
    WebCore::IDBServer::IDBConnectionToClient& ensureConnectionToClient(IPC::Connection::UniqueID, WebCore::IDBConnectionIdentifier);
    void removeConnectionToClient(IPC::Connection::UniqueID);
    void registerConnection(WebCore::IDBServer::UniqueIDBDatabaseConnection&);
    void unregisterConnection(WebCore::IDBServer::UniqueIDBDatabaseConnection&);
    WebCore::IDBServer::UniqueIDBDatabaseConnection* connection(WebCore::IDBDatabaseConnectionIdentifier);
    void registerTransaction(WebCore::IDBServer::UniqueIDBDatabaseTransaction&);
    void unregisterTransaction(WebCore::IDBServer::UniqueIDBDatabaseTransaction&);
    WebCore::IDBServer::UniqueIDBDatabaseTransaction* transaction(WebCore::IDBResourceIdentifier);

private:
    HashMap<WebCore::IDBConnectionIdentifier, std::unique_ptr<IDBStorageConnectionToClient>> m_connectionsToClient;
    HashMap<WebCore::IDBDatabaseConnectionIdentifier, WeakPtr<WebCore::IDBServer::UniqueIDBDatabaseConnection>> m_connections;
    HashMap<WebCore::IDBResourceIdentifier, WeakPtr<WebCore::IDBServer::UniqueIDBDatabaseTransaction>> m_transactions;
};

} // namespace WebKit
