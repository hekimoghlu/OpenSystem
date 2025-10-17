/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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

#include "IDBResourceIdentifier.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class IDBError;
class IDBResultData;

struct IDBDatabaseNameAndVersion;

namespace IDBServer {

class UniqueIDBDatabaseConnection;

class IDBConnectionToClientDelegate : public CanMakeThreadSafeCheckedPtr<IDBConnectionToClientDelegate> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(IDBConnectionToClientDelegate);
public:
    virtual ~IDBConnectionToClientDelegate() = default;
    
    virtual std::optional<IDBConnectionIdentifier> identifier() const = 0;

    virtual void didDeleteDatabase(const IDBResultData&) = 0;
    virtual void didOpenDatabase(const IDBResultData&) = 0;
    virtual void didAbortTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&) = 0;
    virtual void didCommitTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&) = 0;
    virtual void didCreateObjectStore(const IDBResultData&) = 0;
    virtual void didDeleteObjectStore(const IDBResultData&) = 0;
    virtual void didRenameObjectStore(const IDBResultData&) = 0;
    virtual void didClearObjectStore(const IDBResultData&) = 0;
    virtual void didCreateIndex(const IDBResultData&) = 0;
    virtual void didDeleteIndex(const IDBResultData&) = 0;
    virtual void didRenameIndex(const IDBResultData&) = 0;
    virtual void didPutOrAdd(const IDBResultData&) = 0;
    virtual void didGetRecord(const IDBResultData&) = 0;
    virtual void didGetAllRecords(const IDBResultData&) = 0;
    virtual void didGetCount(const IDBResultData&) = 0;
    virtual void didDeleteRecord(const IDBResultData&) = 0;
    virtual void didOpenCursor(const IDBResultData&) = 0;
    virtual void didIterateCursor(const IDBResultData&) = 0;

    virtual void fireVersionChangeEvent(UniqueIDBDatabaseConnection&, const IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion) = 0;
    virtual void didStartTransaction(const IDBResourceIdentifier& transactionIdentifier, const IDBError&) = 0;
    virtual void didCloseFromServer(UniqueIDBDatabaseConnection&, const IDBError&) = 0;
    virtual void notifyOpenDBRequestBlocked(const IDBResourceIdentifier& requestIdentifier, uint64_t oldVersion, uint64_t newVersion) = 0;

    virtual void didGetAllDatabaseNamesAndVersions(const IDBResourceIdentifier&, Vector<IDBDatabaseNameAndVersion>&&) = 0;
};

} // namespace IDBServer
} // namespace WebCore
