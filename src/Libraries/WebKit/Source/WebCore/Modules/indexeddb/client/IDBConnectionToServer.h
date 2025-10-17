/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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

#include "IDBConnectionProxy.h"
#include "IDBConnectionToServerDelegate.h"
#include "IDBDatabaseConnectionIdentifier.h"
#include "IDBIndexIdentifier.h"
#include "IDBObjectStoreIdentifier.h"
#include "IDBResourceIdentifier.h"
#include <pal/SessionID.h>
#include <wtf/CheckedPtr.h>
#include <wtf/Function.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class IDBCursorInfo;
class IDBDatabase;
class IDBError;
class IDBObjectStoreInfo;
class IDBOpenRequestData;
class IDBResultData;
class IDBValue;
class SecurityOrigin;

struct ClientOrigin;
struct IDBDatabaseNameAndVersion;
struct IDBGetAllRecordsData;
struct IDBGetRecordData;
struct IDBIterateCursorData;

namespace IDBClient {

class IDBConnectionToServer final : public ThreadSafeRefCounted<IDBConnectionToServer>, public CanMakeThreadSafeCheckedPtr<IDBConnectionToServer> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(IDBConnectionToServer, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(IDBConnectionToServer);
public:
    WEBCORE_EXPORT static Ref<IDBConnectionToServer> create(IDBConnectionToServerDelegate&, PAL::SessionID);
    WEBCORE_EXPORT ~IDBConnectionToServer();

    WEBCORE_EXPORT IDBConnectionIdentifier identifier() const;

    IDBConnectionProxy& proxy();

    void deleteDatabase(const IDBOpenRequestData&);
    WEBCORE_EXPORT void didDeleteDatabase(const IDBResultData&);

    void openDatabase(const IDBOpenRequestData&);
    WEBCORE_EXPORT void didOpenDatabase(const IDBResultData&);

    void createObjectStore(const IDBRequestData&, const IDBObjectStoreInfo&);
    WEBCORE_EXPORT void didCreateObjectStore(const IDBResultData&);

    void deleteObjectStore(const IDBRequestData&, const String& objectStoreName);
    WEBCORE_EXPORT void didDeleteObjectStore(const IDBResultData&);

    void renameObjectStore(const IDBRequestData&, IDBObjectStoreIdentifier, const String& newName);
    WEBCORE_EXPORT void didRenameObjectStore(const IDBResultData&);

    void clearObjectStore(const IDBRequestData&, IDBObjectStoreIdentifier);
    WEBCORE_EXPORT void didClearObjectStore(const IDBResultData&);

    void createIndex(const IDBRequestData&, const IDBIndexInfo&);
    WEBCORE_EXPORT void didCreateIndex(const IDBResultData&);

    void deleteIndex(const IDBRequestData&, IDBObjectStoreIdentifier, const String& indexName);
    WEBCORE_EXPORT void didDeleteIndex(const IDBResultData&);

    void renameIndex(const IDBRequestData&, IDBObjectStoreIdentifier, IDBIndexIdentifier, const String& newName);
    WEBCORE_EXPORT void didRenameIndex(const IDBResultData&);

    void putOrAdd(const IDBRequestData&, const IDBKeyData&, const IDBValue&, const IndexedDB::ObjectStoreOverwriteMode);
    WEBCORE_EXPORT void didPutOrAdd(const IDBResultData&);

    void getRecord(const IDBRequestData&, const IDBGetRecordData&);
    WEBCORE_EXPORT void didGetRecord(const IDBResultData&);

    void getAllRecords(const IDBRequestData&, const IDBGetAllRecordsData&);
    WEBCORE_EXPORT void didGetAllRecords(const IDBResultData&);

    void getCount(const IDBRequestData&, const IDBKeyRangeData&);
    WEBCORE_EXPORT void didGetCount(const IDBResultData&);

    void deleteRecord(const IDBRequestData&, const IDBKeyRangeData&);
    WEBCORE_EXPORT void didDeleteRecord(const IDBResultData&);

    void openCursor(const IDBRequestData&, const IDBCursorInfo&);
    WEBCORE_EXPORT void didOpenCursor(const IDBResultData&);

    void iterateCursor(const IDBRequestData&, const IDBIterateCursorData&);
    WEBCORE_EXPORT void didIterateCursor(const IDBResultData&);

    void commitTransaction(const IDBResourceIdentifier& transactionIdentifier, uint64_t handledRequestResultsCount);
    WEBCORE_EXPORT void didCommitTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&);

    void didFinishHandlingVersionChangeTransaction(IDBDatabaseConnectionIdentifier, const IDBResourceIdentifier&);

    void abortTransaction(const IDBResourceIdentifier& transactionIdentifier);
    WEBCORE_EXPORT void didAbortTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&);

    WEBCORE_EXPORT void fireVersionChangeEvent(IDBDatabaseConnectionIdentifier, const IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion);
    void didFireVersionChangeEvent(IDBDatabaseConnectionIdentifier, const IDBResourceIdentifier& requestIdentifier, IndexedDB::ConnectionClosedOnBehalfOfServer);

    WEBCORE_EXPORT void didStartTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&);

    WEBCORE_EXPORT void didCloseFromServer(IDBDatabaseConnectionIdentifier, const IDBError&);

    WEBCORE_EXPORT void connectionToServerLost(const IDBError&);

    WEBCORE_EXPORT void notifyOpenDBRequestBlocked(const IDBResourceIdentifier& requestIdentifier, uint64_t oldVersion, uint64_t newVersion);
    void openDBRequestCancelled(const IDBOpenRequestData&);

    void establishTransaction(IDBDatabaseConnectionIdentifier, const IDBTransactionInfo&);

    void databaseConnectionPendingClose(IDBDatabaseConnectionIdentifier);
    void databaseConnectionClosed(IDBDatabaseConnectionIdentifier);

    // To be used when an IDBOpenDBRequest gets a new database connection, optionally with a
    // versionchange transaction, but the page is already torn down.
    void abortOpenAndUpgradeNeeded(IDBDatabaseConnectionIdentifier, const std::optional<IDBResourceIdentifier>& transactionIdentifier);

    void getAllDatabaseNamesAndVersions(const IDBResourceIdentifier&, const ClientOrigin&);
    WEBCORE_EXPORT void didGetAllDatabaseNamesAndVersions(const IDBResourceIdentifier&, Vector<IDBDatabaseNameAndVersion>&&);

private:
    IDBConnectionToServer(IDBConnectionToServerDelegate&, PAL::SessionID);

    typedef void (IDBConnectionToServer::*ResultFunction)(const IDBResultData&);
    void callResultFunctionWithErrorLater(ResultFunction, const IDBResourceIdentifier& requestIdentifier);

    WeakPtr<IDBConnectionToServerDelegate> m_delegate;
    bool m_serverConnectionIsValid { true };

    std::unique_ptr<IDBConnectionProxy> m_proxy;
};

} // namespace IDBClient
} // namespace WebCore
