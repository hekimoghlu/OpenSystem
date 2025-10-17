/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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

#include "IDBConnectionToClientDelegate.h"
#include <wtf/HashSet.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class IDBError;
class IDBResourceIdentifier;
class IDBResultData;

struct IDBDatabaseNameAndVersion;

namespace IDBServer {

class UniqueIDBDatabaseConnection;

class IDBConnectionToClient : public RefCounted<IDBConnectionToClient> {
public:
    WEBCORE_EXPORT static Ref<IDBConnectionToClient> create(IDBConnectionToClientDelegate&);

    IDBConnectionIdentifier identifier() const;

    void didDeleteDatabase(const IDBResultData&);
    void didOpenDatabase(const IDBResultData&);
    void didAbortTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&);
    void didCommitTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&);
    void didCreateObjectStore(const IDBResultData&);
    void didDeleteObjectStore(const IDBResultData&);
    void didRenameObjectStore(const IDBResultData&);
    void didClearObjectStore(const IDBResultData&);
    void didCreateIndex(const IDBResultData&);
    void didDeleteIndex(const IDBResultData&);
    void didRenameIndex(const IDBResultData&);
    void didPutOrAdd(const IDBResultData&);
    void didGetRecord(const IDBResultData&);
    void didGetAllRecords(const IDBResultData&);
    void didGetCount(const IDBResultData&);
    void didDeleteRecord(const IDBResultData&);
    void didOpenCursor(const IDBResultData&);
    void didIterateCursor(const IDBResultData&);

    void fireVersionChangeEvent(UniqueIDBDatabaseConnection&, const IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion);
    void didStartTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&);
    void didCloseFromServer(UniqueIDBDatabaseConnection&, const IDBError&);

    void notifyOpenDBRequestBlocked(const IDBResourceIdentifier& requestIdentifier, uint64_t oldVersion, uint64_t newVersion);

    WEBCORE_EXPORT void didGetAllDatabaseNamesAndVersions(const IDBResourceIdentifier&, Vector<IDBDatabaseNameAndVersion>&&);

    void registerDatabaseConnection(UniqueIDBDatabaseConnection&);
    void unregisterDatabaseConnection(UniqueIDBDatabaseConnection&);
    WEBCORE_EXPORT void connectionToClientClosed();
    bool isClosed() { return m_isClosed; }
    void clearDelegate() { m_delegate = nullptr; }

private:
    IDBConnectionToClient(IDBConnectionToClientDelegate&);
    
    CheckedPtr<IDBConnectionToClientDelegate> m_delegate;
    HashSet<UniqueIDBDatabaseConnection*> m_databaseConnections;
    bool m_isClosed { false };
};

} // namespace IDBServer
} // namespace WebCore
