/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#include "IDBIndexIdentifier.h"
#include "IDBObjectStoreIdentifier.h"
#include "IDBResourceIdentifier.h"
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
namespace IDBClient {
class IDBConnectionToServerDelegate;
}
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::IDBClient::IDBConnectionToServerDelegate> : std::true_type { };
}

namespace WebCore {

class IDBCursorInfo;
class IDBIndexInfo;
class IDBKeyData;
class IDBObjectStoreInfo;
class IDBOpenRequestData;
class IDBRequestData;
class IDBTransactionInfo;
class IDBValue;

struct ClientOrigin;
struct IDBGetAllRecordsData;
struct IDBGetRecordData;
struct IDBIterateCursorData;
class SecurityOriginData;

namespace IndexedDB {
enum class ObjectStoreOverwriteMode : uint8_t;
enum class ConnectionClosedOnBehalfOfServer : bool;
}

struct IDBKeyRangeData;

namespace IDBClient {

class IDBConnectionToServerDelegate : public CanMakeWeakPtr<IDBConnectionToServerDelegate> {
public:
    virtual ~IDBConnectionToServerDelegate() = default;

    virtual std::optional<IDBConnectionIdentifier> identifier() const = 0;
    virtual void deleteDatabase(const IDBOpenRequestData&) = 0;
    virtual void openDatabase(const IDBOpenRequestData&) = 0;
    virtual void abortTransaction(const IDBResourceIdentifier&) = 0;
    virtual void commitTransaction(const IDBResourceIdentifier&, uint64_t handledRequestResultsCount) = 0;
    virtual void didFinishHandlingVersionChangeTransaction(IDBDatabaseConnectionIdentifier, const IDBResourceIdentifier&) = 0;
    virtual void createObjectStore(const IDBRequestData&, const IDBObjectStoreInfo&) = 0;
    virtual void deleteObjectStore(const IDBRequestData&, const String& objectStoreName) = 0;
    virtual void renameObjectStore(const IDBRequestData&, IDBObjectStoreIdentifier, const String& newName) = 0;
    virtual void clearObjectStore(const IDBRequestData&, IDBObjectStoreIdentifier) = 0;
    virtual void createIndex(const IDBRequestData&, const IDBIndexInfo&) = 0;
    virtual void deleteIndex(const IDBRequestData&, IDBObjectStoreIdentifier, const String& indexName) = 0;
    virtual void renameIndex(const IDBRequestData&, IDBObjectStoreIdentifier, IDBIndexIdentifier, const String& newName) = 0;
    virtual void putOrAdd(const IDBRequestData&, const IDBKeyData&, const IDBValue&, const IndexedDB::ObjectStoreOverwriteMode) = 0;
    virtual void getRecord(const IDBRequestData&, const IDBGetRecordData&) = 0;
    virtual void getAllRecords(const IDBRequestData&, const IDBGetAllRecordsData&) = 0;
    virtual void getCount(const IDBRequestData&, const IDBKeyRangeData&) = 0;
    virtual void deleteRecord(const IDBRequestData&, const IDBKeyRangeData&) = 0;
    virtual void openCursor(const IDBRequestData&, const IDBCursorInfo&) = 0;
    virtual void iterateCursor(const IDBRequestData&, const IDBIterateCursorData&) = 0;

    virtual void establishTransaction(IDBDatabaseConnectionIdentifier, const IDBTransactionInfo&) = 0;
    virtual void databaseConnectionPendingClose(IDBDatabaseConnectionIdentifier) = 0;
    virtual void databaseConnectionClosed(IDBDatabaseConnectionIdentifier) = 0;
    virtual void abortOpenAndUpgradeNeeded(IDBDatabaseConnectionIdentifier, const std::optional<IDBResourceIdentifier>& transactionIdentifier) = 0;
    virtual void didFireVersionChangeEvent(IDBDatabaseConnectionIdentifier, const IDBResourceIdentifier& requestIdentifier, const IndexedDB::ConnectionClosedOnBehalfOfServer) = 0;
    virtual void openDBRequestCancelled(const IDBOpenRequestData&) = 0;

    virtual void getAllDatabaseNamesAndVersions(const IDBResourceIdentifier&, const ClientOrigin&) = 0;
};

} // namespace IDBClient
} // namespace WebCore
