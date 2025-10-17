/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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
#include "IDBResultData.h"

#include "UniqueIDBDatabase.h"
#include "UniqueIDBDatabaseConnection.h"
#include "UniqueIDBDatabaseTransaction.h"

namespace WebCore {

IDBResultData::IDBResultData()
{
}

IDBResultData::IDBResultData(const IDBResourceIdentifier& requestIdentifier)
    : m_requestIdentifier(requestIdentifier)
{
}

IDBResultData::IDBResultData(IDBResultType type, const IDBResourceIdentifier& requestIdentifier)
    : m_type(type)
    , m_requestIdentifier(requestIdentifier)
{
}

IDBResultData::IDBResultData(const IDBResultData& other)
    : m_type(other.m_type)
    , m_requestIdentifier(other.m_requestIdentifier)
    , m_error(other.m_error)
    , m_databaseConnectionIdentifier(other.m_databaseConnectionIdentifier)
    , m_resultInteger(other.m_resultInteger)
{
    if (other.m_databaseInfo)
        m_databaseInfo = makeUnique<IDBDatabaseInfo>(*other.m_databaseInfo);
    if (other.m_transactionInfo)
        m_transactionInfo = makeUnique<IDBTransactionInfo>(*other.m_transactionInfo);
    if (other.m_resultKey)
        m_resultKey = makeUnique<IDBKeyData>(*other.m_resultKey);
    if (other.m_getResult)
        m_getResult = makeUnique<IDBGetResult>(*other.m_getResult);
    if (other.m_getAllResult)
        m_getAllResult = makeUnique<IDBGetAllResult>(*other.m_getAllResult);
}

IDBResultData::IDBResultData(const IDBResultData& that, IsolatedCopyTag)
{
    isolatedCopy(that, *this);
}

IDBResultData IDBResultData::isolatedCopy() const
{
    return { *this, IsolatedCopy };
}

void IDBResultData::isolatedCopy(const IDBResultData& source, IDBResultData& destination)
{
    destination.m_type = source.m_type;
    destination.m_requestIdentifier = source.m_requestIdentifier.isolatedCopy();
    destination.m_error = source.m_error.isolatedCopy();
    destination.m_databaseConnectionIdentifier = source.m_databaseConnectionIdentifier;
    destination.m_resultInteger = source.m_resultInteger;

    if (source.m_databaseInfo)
        destination.m_databaseInfo = makeUnique<IDBDatabaseInfo>(*source.m_databaseInfo, IDBDatabaseInfo::IsolatedCopy);
    if (source.m_transactionInfo)
        destination.m_transactionInfo = makeUnique<IDBTransactionInfo>(*source.m_transactionInfo, IDBTransactionInfo::IsolatedCopy);
    if (source.m_resultKey)
        destination.m_resultKey = makeUnique<IDBKeyData>(*source.m_resultKey, IDBKeyData::IsolatedCopy);
    if (source.m_getResult)
        destination.m_getResult = makeUnique<IDBGetResult>(*source.m_getResult, IDBGetResult::IsolatedCopy);
    if (source.m_getAllResult)
        destination.m_getAllResult = makeUnique<IDBGetAllResult>(*source.m_getAllResult, IDBGetAllResult::IsolatedCopy);
}

IDBResultData IDBResultData::error(const IDBResourceIdentifier& requestIdentifier, const IDBError& error)
{
    IDBResultData result { requestIdentifier };
    result.m_type = IDBResultType::Error;
    result.m_error = error;
    return result;
}

IDBResultData IDBResultData::openDatabaseSuccess(const IDBResourceIdentifier& requestIdentifier, IDBServer::UniqueIDBDatabaseConnection& connection)
{
    IDBResultData result { requestIdentifier };
    result.m_type = IDBResultType::OpenDatabaseSuccess;
    result.m_databaseConnectionIdentifier = connection.identifier();
    result.m_databaseInfo = makeUnique<IDBDatabaseInfo>(connection.database()->info());
    return result;
}


IDBResultData IDBResultData::openDatabaseUpgradeNeeded(const IDBResourceIdentifier& requestIdentifier, IDBServer::UniqueIDBDatabaseTransaction& transaction, IDBServer::UniqueIDBDatabaseConnection& connection)
{
    IDBResultData result { requestIdentifier };
    result.m_type = IDBResultType::OpenDatabaseUpgradeNeeded;
    result.m_databaseConnectionIdentifier = connection.identifier();
    result.m_databaseInfo = makeUnique<IDBDatabaseInfo>(connection.database()->info());
    result.m_transactionInfo = makeUnique<IDBTransactionInfo>(transaction.info());
    return result;
}

IDBResultData IDBResultData::deleteDatabaseSuccess(const IDBResourceIdentifier& requestIdentifier, const IDBDatabaseInfo& info)
{
    IDBResultData result {IDBResultType::DeleteDatabaseSuccess, requestIdentifier };
    result.m_databaseInfo = makeUnique<IDBDatabaseInfo>(info);
    return result;
}

IDBResultData IDBResultData::createObjectStoreSuccess(const IDBResourceIdentifier& requestIdentifier)
{
    return { IDBResultType::CreateObjectStoreSuccess, requestIdentifier };
}

IDBResultData IDBResultData::deleteObjectStoreSuccess(const IDBResourceIdentifier& requestIdentifier)
{
    return { IDBResultType::DeleteObjectStoreSuccess, requestIdentifier };
}

IDBResultData IDBResultData::renameObjectStoreSuccess(const IDBResourceIdentifier& requestIdentifier)
{
    return { IDBResultType::RenameObjectStoreSuccess, requestIdentifier };
}

IDBResultData IDBResultData::clearObjectStoreSuccess(const IDBResourceIdentifier& requestIdentifier)
{
    return { IDBResultType::ClearObjectStoreSuccess, requestIdentifier };
}

IDBResultData IDBResultData::createIndexSuccess(const IDBResourceIdentifier& requestIdentifier)
{
    return { IDBResultType::CreateIndexSuccess, requestIdentifier };
}

IDBResultData IDBResultData::deleteIndexSuccess(const IDBResourceIdentifier& requestIdentifier)
{
    return { IDBResultType::DeleteIndexSuccess, requestIdentifier };
}

IDBResultData IDBResultData::renameIndexSuccess(const IDBResourceIdentifier& requestIdentifier)
{
    return { IDBResultType::RenameIndexSuccess, requestIdentifier };
}

IDBResultData IDBResultData::putOrAddSuccess(const IDBResourceIdentifier& requestIdentifier, const IDBKeyData& resultKey)
{
    IDBResultData result { IDBResultType::PutOrAddSuccess, requestIdentifier };
    result.m_resultKey = makeUnique<IDBKeyData>(resultKey);
    return result;
}

IDBResultData IDBResultData::getRecordSuccess(const IDBResourceIdentifier& requestIdentifier, const IDBGetResult& getResult)
{
    IDBResultData result { IDBResultType::GetRecordSuccess, requestIdentifier };
    result.m_getResult = makeUnique<IDBGetResult>(getResult);
    return result;
}

IDBResultData IDBResultData::getAllRecordsSuccess(const IDBResourceIdentifier& requestIdentifier, const IDBGetAllResult& getAllResult)
{
    IDBResultData result { IDBResultType::GetAllRecordsSuccess, requestIdentifier };
    result.m_getAllResult = makeUnique<IDBGetAllResult>(getAllResult);
    return result;
}

IDBResultData IDBResultData::getCountSuccess(const IDBResourceIdentifier& requestIdentifier, uint64_t count)
{
    IDBResultData result { IDBResultType::GetRecordSuccess, requestIdentifier };
    result.m_resultInteger = count;
    return result;
}

IDBResultData IDBResultData::deleteRecordSuccess(const IDBResourceIdentifier& requestIdentifier)
{
    return { IDBResultType::DeleteRecordSuccess, requestIdentifier };
}

IDBResultData IDBResultData::openCursorSuccess(const IDBResourceIdentifier& requestIdentifier, const IDBGetResult& getResult)
{
    IDBResultData result { IDBResultType::OpenCursorSuccess, requestIdentifier };
    result.m_getResult = makeUnique<IDBGetResult>(getResult);
    return result;
}

IDBResultData IDBResultData::iterateCursorSuccess(const IDBResourceIdentifier& requestIdentifier, const IDBGetResult& getResult)
{
    IDBResultData result { IDBResultType::IterateCursorSuccess, requestIdentifier };
    result.m_getResult = makeUnique<IDBGetResult>(getResult);
    return result;
}

const IDBDatabaseInfo& IDBResultData::databaseInfo() const
{
    RELEASE_ASSERT(m_databaseInfo);
    return *m_databaseInfo;
}

const IDBTransactionInfo& IDBResultData::transactionInfo() const
{
    RELEASE_ASSERT(m_transactionInfo);
    return *m_transactionInfo;
}

const IDBGetResult& IDBResultData::getResult() const
{
    RELEASE_ASSERT(m_getResult);
    return *m_getResult;
}

IDBGetResult& IDBResultData::getResultRef()
{
    RELEASE_ASSERT(m_getResult);
    return *m_getResult;
}

const IDBGetAllResult& IDBResultData::getAllResult() const
{
    RELEASE_ASSERT(m_getAllResult);
    return *m_getAllResult;
}

} // namespace WebCore
