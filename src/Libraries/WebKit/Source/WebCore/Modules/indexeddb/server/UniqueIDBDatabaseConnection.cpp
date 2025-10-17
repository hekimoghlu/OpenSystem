/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
#include "UniqueIDBDatabaseConnection.h"

#include "IDBConnectionToClient.h"
#include "IDBTransactionInfo.h"
#include "Logging.h"
#include "ServerOpenDBRequest.h"
#include "UniqueIDBDatabase.h"
#include "UniqueIDBDatabaseManager.h"

namespace WebCore {
namespace IDBServer {

Ref<UniqueIDBDatabaseConnection> UniqueIDBDatabaseConnection::create(UniqueIDBDatabase& database, ServerOpenDBRequest& request)
{
    return adoptRef(*new UniqueIDBDatabaseConnection(database, request));
}

UniqueIDBDatabaseConnection::UniqueIDBDatabaseConnection(UniqueIDBDatabase& database, ServerOpenDBRequest& request)
    : m_database(database)
    , m_manager(database.manager())
    , m_connectionToClient(request.connection())
    , m_openRequestIdentifier(request.requestData().requestIdentifier())
{
    if (auto* manager = database.manager()) {
        m_manager = *manager;
        m_manager->registerConnection(*this);
    }
    m_connectionToClient->registerDatabaseConnection(*this);
}

UniqueIDBDatabaseConnection::~UniqueIDBDatabaseConnection()
{
    ASSERT(m_transactionMap.isEmpty());

    if (m_manager)
        m_manager->unregisterConnection(*this);
    m_connectionToClient->unregisterDatabaseConnection(*this);
}

UniqueIDBDatabaseManager* UniqueIDBDatabaseConnection::manager()
{
    return m_manager.get();
}

bool UniqueIDBDatabaseConnection::hasNonFinishedTransactions() const
{
    return !m_transactionMap.isEmpty();
}

void UniqueIDBDatabaseConnection::abortTransactionWithoutCallback(UniqueIDBDatabaseTransaction& transaction)
{
    ASSERT(m_transactionMap.contains(transaction.info().identifier()));
    ASSERT(m_database);

    const auto& transactionIdentifier = transaction.info().identifier();
    m_database->abortTransaction(transaction, [this, weakThis = WeakPtr { *this }, transactionIdentifier](const IDBError&) {
        if (!weakThis)
            return;
        ASSERT(m_transactionMap.contains(transactionIdentifier));
        m_transactionMap.remove(transactionIdentifier);
    });
}

void UniqueIDBDatabaseConnection::connectionPendingCloseFromClient()
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::connectionPendingCloseFromClient - %s - %" PRIu64, m_openRequestIdentifier.loggingString().utf8().data(), identifier().toUInt64());

    m_closePending = true;
}

void UniqueIDBDatabaseConnection::connectionClosedFromClient()
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::connectionClosedFromClient - %s - %" PRIu64, m_openRequestIdentifier.loggingString().utf8().data(), identifier().toUInt64());

    ASSERT(m_database);
    m_database->connectionClosedFromClient(*this);
}

void UniqueIDBDatabaseConnection::didFireVersionChangeEvent(const IDBResourceIdentifier& requestIdentifier, IndexedDB::ConnectionClosedOnBehalfOfServer connectionClosed)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::didFireVersionChangeEvent - %s - %" PRIu64, m_openRequestIdentifier.loggingString().utf8().data(), identifier().toUInt64());

    ASSERT(m_database);
    m_database->didFireVersionChangeEvent(*this, requestIdentifier, connectionClosed);
}

void UniqueIDBDatabaseConnection::didFinishHandlingVersionChange(const IDBResourceIdentifier& transactionIdentifier)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::didFinishHandlingVersionChange - %s - %" PRIu64, transactionIdentifier.loggingString().utf8().data(), identifier().toUInt64());

    ASSERT(m_database);
    m_database->didFinishHandlingVersionChange(*this, transactionIdentifier);
}

void UniqueIDBDatabaseConnection::fireVersionChangeEvent(const IDBResourceIdentifier& requestIdentifier, uint64_t requestedVersion)
{
    ASSERT(!m_closePending);
    m_connectionToClient->fireVersionChangeEvent(*this, requestIdentifier, requestedVersion);
}

UniqueIDBDatabaseTransaction& UniqueIDBDatabaseConnection::createVersionChangeTransaction(uint64_t newVersion)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::createVersionChangeTransaction - %s - %" PRIu64, m_openRequestIdentifier.loggingString().utf8().data(), identifier().toUInt64());
    ASSERT(!m_closePending);

    IDBTransactionInfo info = IDBTransactionInfo::versionChange(m_connectionToClient, m_database->info(), newVersion);

    Ref<UniqueIDBDatabaseTransaction> transaction = UniqueIDBDatabaseTransaction::create(*this, info);
    m_transactionMap.set(transaction->info().identifier(), &transaction.get());

    return transaction.get();
}

void UniqueIDBDatabaseConnection::establishTransaction(const IDBTransactionInfo& info)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::establishTransaction - %s - %" PRIu64, m_openRequestIdentifier.loggingString().utf8().data(), identifier().toUInt64());

    ASSERT(info.mode() != IDBTransactionMode::Versionchange);

    // No transactions should ever come from the client after the client has already told us
    // the connection is closing.
    ASSERT(!m_closePending);

    Ref<UniqueIDBDatabaseTransaction> transaction = UniqueIDBDatabaseTransaction::create(*this, info);
    m_transactionMap.set(transaction->info().identifier(), &transaction.get());

    ASSERT(m_database);
    m_database->enqueueTransaction(WTFMove(transaction));
}

void UniqueIDBDatabaseConnection::didAbortTransaction(UniqueIDBDatabaseTransaction& transaction, const IDBError& error)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::didAbortTransaction - %s - %" PRIu64, m_openRequestIdentifier.loggingString().utf8().data(), identifier().toUInt64());

    auto transactionIdentifier = transaction.info().identifier();
    auto takenTransaction = m_transactionMap.take(transactionIdentifier);
    ASSERT(takenTransaction);

    m_connectionToClient->didAbortTransaction(transactionIdentifier, error);
}

void UniqueIDBDatabaseConnection::didCommitTransaction(UniqueIDBDatabaseTransaction& transaction, const IDBError& error)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::didCommitTransaction - %s - %" PRIu64, m_openRequestIdentifier.loggingString().utf8().data(), identifier().toUInt64());

    auto transactionIdentifier = transaction.info().identifier();

    ASSERT(m_transactionMap.contains(transactionIdentifier) || !error.isNull());
    m_transactionMap.remove(transactionIdentifier);

    m_connectionToClient->didCommitTransaction(transactionIdentifier, error);
}

void UniqueIDBDatabaseConnection::didCreateObjectStore(const IDBResultData& resultData)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::didCreateObjectStore");

    m_connectionToClient->didCreateObjectStore(resultData);
}

void UniqueIDBDatabaseConnection::didDeleteObjectStore(const IDBResultData& resultData)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::didDeleteObjectStore");

    m_connectionToClient->didDeleteObjectStore(resultData);
}

void UniqueIDBDatabaseConnection::didRenameObjectStore(const IDBResultData& resultData)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::didRenameObjectStore");

    m_connectionToClient->didRenameObjectStore(resultData);
}

void UniqueIDBDatabaseConnection::didClearObjectStore(const IDBResultData& resultData)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::didClearObjectStore");

    m_connectionToClient->didClearObjectStore(resultData);
}

void UniqueIDBDatabaseConnection::didCreateIndex(const IDBResultData& resultData)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::didCreateIndex");

    m_connectionToClient->didCreateIndex(resultData);
}

void UniqueIDBDatabaseConnection::didDeleteIndex(const IDBResultData& resultData)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::didDeleteIndex");

    m_connectionToClient->didDeleteIndex(resultData);
}

void UniqueIDBDatabaseConnection::didRenameIndex(const IDBResultData& resultData)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::didRenameIndex");

    m_connectionToClient->didRenameIndex(resultData);
}

bool UniqueIDBDatabaseConnection::connectionIsClosing() const
{
    return m_closePending;
}

void UniqueIDBDatabaseConnection::deleteTransaction(UniqueIDBDatabaseTransaction& transaction)
{
    LOG(IndexedDB, "UniqueIDBDatabaseConnection::deleteTransaction - %s", transaction.info().loggingString().utf8().data());
    
    auto transactionIdentifier = transaction.info().identifier();
    
    ASSERT(m_transactionMap.contains(transactionIdentifier));
    m_transactionMap.remove(transactionIdentifier);
}

} // namespace IDBServer
} // namespace WebCore
