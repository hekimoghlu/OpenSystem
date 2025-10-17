/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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
#include "IDBDatabaseInfo.h"
#include "IDBError.h"
#include "IDBGetAllResult.h"
#include "IDBGetResult.h"
#include "IDBKeyData.h"
#include "IDBResourceIdentifier.h"
#include "IDBTransactionInfo.h"
#include "ThreadSafeDataBuffer.h"
#include <wtf/ArgumentCoder.h>

namespace WebCore {

class ThreadSafeDataBuffer;

enum class IDBResultType : uint8_t {
    Error,
    OpenDatabaseSuccess,
    OpenDatabaseUpgradeNeeded,
    DeleteDatabaseSuccess,
    CreateObjectStoreSuccess,
    DeleteObjectStoreSuccess,
    ClearObjectStoreSuccess,
    PutOrAddSuccess,
    GetRecordSuccess,
    GetAllRecordsSuccess,
    GetCountSuccess,
    DeleteRecordSuccess,
    CreateIndexSuccess,
    DeleteIndexSuccess,
    OpenCursorSuccess,
    IterateCursorSuccess,
    RenameObjectStoreSuccess,
    RenameIndexSuccess,
};

namespace IDBServer {
class UniqueIDBDatabaseConnection;
class UniqueIDBDatabaseTransaction;
}

class IDBResultData {
public:
    static IDBResultData error(const IDBResourceIdentifier&, const IDBError&);
    static IDBResultData openDatabaseSuccess(const IDBResourceIdentifier&, IDBServer::UniqueIDBDatabaseConnection&);
    static IDBResultData openDatabaseUpgradeNeeded(const IDBResourceIdentifier&, IDBServer::UniqueIDBDatabaseTransaction&, IDBServer::UniqueIDBDatabaseConnection&);
    static IDBResultData deleteDatabaseSuccess(const IDBResourceIdentifier&, const IDBDatabaseInfo&);
    static IDBResultData createObjectStoreSuccess(const IDBResourceIdentifier&);
    static IDBResultData deleteObjectStoreSuccess(const IDBResourceIdentifier&);
    static IDBResultData renameObjectStoreSuccess(const IDBResourceIdentifier&);
    static IDBResultData clearObjectStoreSuccess(const IDBResourceIdentifier&);
    static IDBResultData createIndexSuccess(const IDBResourceIdentifier&);
    static IDBResultData deleteIndexSuccess(const IDBResourceIdentifier&);
    static IDBResultData renameIndexSuccess(const IDBResourceIdentifier&);
    static IDBResultData putOrAddSuccess(const IDBResourceIdentifier&, const IDBKeyData&);
    static IDBResultData getRecordSuccess(const IDBResourceIdentifier&, const IDBGetResult&);
    static IDBResultData getAllRecordsSuccess(const IDBResourceIdentifier&, const IDBGetAllResult&);
    static IDBResultData getCountSuccess(const IDBResourceIdentifier&, uint64_t count);
    static IDBResultData deleteRecordSuccess(const IDBResourceIdentifier&);
    static IDBResultData openCursorSuccess(const IDBResourceIdentifier&, const IDBGetResult&);
    static IDBResultData iterateCursorSuccess(const IDBResourceIdentifier&, const IDBGetResult&);

    WEBCORE_EXPORT IDBResultData(const IDBResultData&);
    IDBResultData(IDBResultData&&) = default;
    IDBResultData& operator=(IDBResultData&&) = default;

    enum IsolatedCopyTag { IsolatedCopy };
    IDBResultData(const IDBResultData&, IsolatedCopyTag);
    WEBCORE_EXPORT IDBResultData isolatedCopy() const;

    IDBResultType type() const { return m_type; }
    IDBResourceIdentifier requestIdentifier() const { return m_requestIdentifier; }

    const IDBError& error() const { return m_error; }
    IDBDatabaseConnectionIdentifier databaseConnectionIdentifier() const { return *m_databaseConnectionIdentifier; }

    const IDBDatabaseInfo& databaseInfo() const;
    const IDBTransactionInfo& transactionInfo() const;

    const IDBKeyData* resultKey() const { return m_resultKey.get(); }
    uint64_t resultInteger() const { return m_resultInteger; }

    WEBCORE_EXPORT const IDBGetResult& getResult() const;
    WEBCORE_EXPORT IDBGetResult& getResultRef();
    WEBCORE_EXPORT const IDBGetAllResult& getAllResult() const;

    WEBCORE_EXPORT IDBResultData();

private:
    friend struct IPC::ArgumentCoder<IDBResultData, void>;

    IDBResultData(const IDBResourceIdentifier&);
    IDBResultData(IDBResultType, const IDBResourceIdentifier&);

    static void isolatedCopy(const IDBResultData& source, IDBResultData& destination);

    IDBResultType m_type { IDBResultType::Error };
    IDBResourceIdentifier m_requestIdentifier;

    IDBError m_error;
    std::optional<IDBDatabaseConnectionIdentifier> m_databaseConnectionIdentifier;
    std::unique_ptr<IDBDatabaseInfo> m_databaseInfo;
    std::unique_ptr<IDBTransactionInfo> m_transactionInfo;
    std::unique_ptr<IDBKeyData> m_resultKey;
    std::unique_ptr<IDBGetResult> m_getResult;
    std::unique_ptr<IDBGetAllResult> m_getAllResult;
    uint64_t m_resultInteger { 0 };
};

} // namespace WebCore

