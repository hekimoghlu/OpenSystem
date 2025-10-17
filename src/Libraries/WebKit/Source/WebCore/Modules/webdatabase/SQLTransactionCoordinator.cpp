/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
#include "SQLTransactionCoordinator.h"

#include "Database.h"
#include "SQLTransaction.h"
#include "SecurityOrigin.h"
#include "SecurityOriginData.h"
#include <wtf/Deque.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static String getDatabaseIdentifier(SQLTransaction& transaction)
{
    return transaction.database().securityOrigin().databaseIdentifier();
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(SQLTransactionCoordinator);

SQLTransactionCoordinator::SQLTransactionCoordinator()
    : m_isShuttingDown(false)
{
}

void SQLTransactionCoordinator::processPendingTransactions(CoordinationInfo& info)
{
    if (info.activeWriteTransaction || info.pendingTransactions.isEmpty())
        return;

    RefPtr<SQLTransaction> firstPendingTransaction = info.pendingTransactions.first();
    if (firstPendingTransaction->isReadOnly()) {
        do {
            firstPendingTransaction = info.pendingTransactions.takeFirst();
            info.activeReadTransactions.add(firstPendingTransaction);
            firstPendingTransaction->lockAcquired();
        } while (!info.pendingTransactions.isEmpty() && info.pendingTransactions.first()->isReadOnly());
    } else if (info.activeReadTransactions.isEmpty()) {
        info.pendingTransactions.removeFirst();
        info.activeWriteTransaction = firstPendingTransaction;
        firstPendingTransaction->lockAcquired();
    }
}

void SQLTransactionCoordinator::acquireLock(SQLTransaction& transaction)
{
    ASSERT(!m_isShuttingDown);

    String dbIdentifier = getDatabaseIdentifier(transaction);

    CoordinationInfoMap::iterator coordinationInfoIterator = m_coordinationInfoMap.find(dbIdentifier);
    if (coordinationInfoIterator == m_coordinationInfoMap.end()) {
        // No pending transactions for this DB
        coordinationInfoIterator = m_coordinationInfoMap.add(dbIdentifier, CoordinationInfo()).iterator;
    }

    CoordinationInfo& info = coordinationInfoIterator->value;
    info.pendingTransactions.append(&transaction);
    processPendingTransactions(info);
}

void SQLTransactionCoordinator::releaseLock(SQLTransaction& transaction)
{
    if (m_isShuttingDown)
        return;

    String dbIdentifier = getDatabaseIdentifier(transaction);

    CoordinationInfoMap::iterator coordinationInfoIterator = m_coordinationInfoMap.find(dbIdentifier);
    ASSERT(coordinationInfoIterator != m_coordinationInfoMap.end());
    CoordinationInfo& info = coordinationInfoIterator->value;

    if (transaction.isReadOnly()) {
        ASSERT(info.activeReadTransactions.contains(&transaction));
        info.activeReadTransactions.remove(&transaction);
    } else {
        ASSERT(info.activeWriteTransaction == &transaction);
        info.activeWriteTransaction = nullptr;
    }

    processPendingTransactions(info);
}

void SQLTransactionCoordinator::shutdown()
{
    // Prevent releaseLock() from accessing / changing the coordinationInfo
    // while we're shutting down.
    m_isShuttingDown = true;

    // Notify all transactions in progress that the database thread is shutting down
    for (auto& info : m_coordinationInfoMap.values()) {
        // Clean up transactions that have reached "lockAcquired":
        // Transaction phase 4 cleanup. See comment on "What happens if a
        // transaction is interrupted?" at the top of SQLTransactionBackend.cpp.
        if (info.activeWriteTransaction)
            info.activeWriteTransaction->notifyDatabaseThreadIsShuttingDown();
        for (auto& transaction : info.activeReadTransactions)
            transaction->notifyDatabaseThreadIsShuttingDown();

        // Clean up transactions that have NOT reached "lockAcquired":
        // Transaction phase 3 cleanup. See comment on "What happens if a
        // transaction is interrupted?" at the top of SQLTransactionBackend.cpp.
        while (!info.pendingTransactions.isEmpty()) {
            RefPtr transaction = info.pendingTransactions.takeFirst();
            transaction->notifyDatabaseThreadIsShuttingDown();
        }
    }

    // Clean up all pending transactions for all databases
    m_coordinationInfoMap.clear();
}

} // namespace WebCore
