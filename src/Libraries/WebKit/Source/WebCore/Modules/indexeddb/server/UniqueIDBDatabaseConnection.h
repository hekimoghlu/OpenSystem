/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 9, 2022.
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

#include "IDBDatabaseConnectionIdentifier.h"
#include "UniqueIDBDatabase.h"
#include <wtf/HashMap.h>
#include <wtf/Identified.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebCore {

class IDBError;
class IDBResultData;
class UniqueIDBDatabaseManager;

namespace IDBServer {

class IDBConnectionToClient;
class ServerOpenDBRequest;
class UniqueIDBDatabase;
class UniqueIDBDatabaseTransaction;

class UniqueIDBDatabaseConnection : public RefCountedAndCanMakeWeakPtr<UniqueIDBDatabaseConnection>, public Identified<IDBDatabaseConnectionIdentifier> {
public:
    static Ref<UniqueIDBDatabaseConnection> create(UniqueIDBDatabase&, ServerOpenDBRequest&);

    WEBCORE_EXPORT ~UniqueIDBDatabaseConnection();

    const IDBResourceIdentifier& openRequestIdentifier() { return m_openRequestIdentifier; }
    UniqueIDBDatabase* database() { return m_database.get(); }
    UniqueIDBDatabaseManager* manager();
    IDBConnectionToClient& connectionToClient() { return m_connectionToClient; }

    WEBCORE_EXPORT void connectionPendingCloseFromClient();
    WEBCORE_EXPORT void connectionClosedFromClient();

    bool closePending() const { return m_closePending; }

    bool hasNonFinishedTransactions() const;

    void fireVersionChangeEvent(const IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion);
    UniqueIDBDatabaseTransaction& createVersionChangeTransaction(uint64_t newVersion);

    WEBCORE_EXPORT void establishTransaction(const IDBTransactionInfo&);
    void didAbortTransaction(UniqueIDBDatabaseTransaction&, const IDBError&);
    void didCommitTransaction(UniqueIDBDatabaseTransaction&, const IDBError&);
    void didCreateObjectStore(const IDBResultData&);
    void didDeleteObjectStore(const IDBResultData&);
    void didRenameObjectStore(const IDBResultData&);
    void didClearObjectStore(const IDBResultData&);
    void didCreateIndex(const IDBResultData&);
    void didDeleteIndex(const IDBResultData&);
    void didRenameIndex(const IDBResultData&);
    WEBCORE_EXPORT void didFireVersionChangeEvent(const IDBResourceIdentifier& requestIdentifier, IndexedDB::ConnectionClosedOnBehalfOfServer);
    WEBCORE_EXPORT void didFinishHandlingVersionChange(const IDBResourceIdentifier& transactionIdentifier);

    void abortTransactionWithoutCallback(UniqueIDBDatabaseTransaction&);

    bool connectionIsClosing() const;

    void deleteTransaction(UniqueIDBDatabaseTransaction&);

private:
    UniqueIDBDatabaseConnection(UniqueIDBDatabase&, ServerOpenDBRequest&);

    WeakPtr<UniqueIDBDatabase> m_database;
    WeakPtr<UniqueIDBDatabaseManager> m_manager;
    Ref<IDBConnectionToClient> m_connectionToClient;
    IDBResourceIdentifier m_openRequestIdentifier;

    bool m_closePending { false };

    HashMap<IDBResourceIdentifier, RefPtr<UniqueIDBDatabaseTransaction>> m_transactionMap;
};

} // namespace IDBServer
} // namespace WebCore
