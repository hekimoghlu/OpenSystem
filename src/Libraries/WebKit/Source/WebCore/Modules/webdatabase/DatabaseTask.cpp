/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 8, 2022.
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
#include "DatabaseTask.h"

#include "Database.h"
#include "Logging.h"
#include "SQLTransaction.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

DatabaseTaskSynchronizer::DatabaseTaskSynchronizer()
{
}

void DatabaseTaskSynchronizer::waitForTaskCompletion()
{
    Locker locker { m_synchronousLock };
    while (!m_taskCompleted)
        m_synchronousCondition.wait(m_synchronousLock);
}

void DatabaseTaskSynchronizer::taskCompleted()
{
    Locker locker { m_synchronousLock };
    m_taskCompleted = true;
    m_synchronousCondition.notifyOne();
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(DatabaseTask);

DatabaseTask::DatabaseTask(Database& database, DatabaseTaskSynchronizer* synchronizer)
    : m_database(database)
    , m_synchronizer(synchronizer)
{
}

DatabaseTask::~DatabaseTask()
{
    ASSERT(m_complete || !m_synchronizer);
}

void DatabaseTask::performTask()
{
    // Database tasks are meant to be used only once, so make sure this one hasn't been performed before.
    ASSERT(!m_complete);

    LOG(StorageAPI, "Performing %s %p\n", debugTaskName().characters(), this);

    m_database.resetAuthorizer();

    doPerformTask();

    if (m_synchronizer)
        m_synchronizer->taskCompleted();

#if ASSERT_ENABLED
    m_complete = true;
#endif
}

// *** DatabaseOpenTask ***
// Opens the database file and verifies the version matches the expected version.

DatabaseOpenTask::DatabaseOpenTask(Database& database, bool setVersionInNewDatabase, DatabaseTaskSynchronizer& synchronizer, ExceptionOr<void>& result)
    : DatabaseTask(database, &synchronizer)
    , m_setVersionInNewDatabase(setVersionInNewDatabase)
    , m_result(result)
{
}

void DatabaseOpenTask::doPerformTask()
{
    m_result = crossThreadCopy(database().performOpenAndVerify(m_setVersionInNewDatabase));
}

#if !LOG_DISABLED

ASCIILiteral DatabaseOpenTask::debugTaskName() const
{
    return "DatabaseOpenTask"_s;
}

#endif

// *** DatabaseCloseTask ***
// Closes the database.

DatabaseCloseTask::DatabaseCloseTask(Database& database, DatabaseTaskSynchronizer& synchronizer)
    : DatabaseTask(database, &synchronizer)
{
}

void DatabaseCloseTask::doPerformTask()
{
    database().performClose();
}

#if !LOG_DISABLED

ASCIILiteral DatabaseCloseTask::debugTaskName() const
{
    return "DatabaseCloseTask"_s;
}

#endif

// *** DatabaseTransactionTask ***
// Starts a transaction that will report its results via a callback.

DatabaseTransactionTask::DatabaseTransactionTask(RefPtr<SQLTransaction>&& transaction)
    : DatabaseTask(transaction->database(), 0)
    , m_transaction(WTFMove(transaction))
    , m_didPerformTask(false)
{
}

DatabaseTransactionTask::~DatabaseTransactionTask()
{
    // If the task is being destructed without the transaction ever being run,
    // then we must either have an error or an interruption. Give the
    // transaction a chance to clean up since it may not have been able to
    // run to its clean up state.

    // Transaction phase 2 cleanup. See comment on "What happens if a
    // transaction is interrupted?" at the top of SQLTransactionBackend.cpp.

    if (!m_didPerformTask)
        m_transaction->notifyDatabaseThreadIsShuttingDown();
}

void DatabaseTransactionTask::doPerformTask()
{
    m_transaction->performNextStep();
    m_didPerformTask = true;
}

#if !LOG_DISABLED

ASCIILiteral DatabaseTransactionTask::debugTaskName() const
{
    return "DatabaseTransactionTask"_s;
}

#endif

// *** DatabaseTableNamesTask ***
// Retrieves a list of all tables in the database - for WebInspector support.

DatabaseTableNamesTask::DatabaseTableNamesTask(Database& database, DatabaseTaskSynchronizer& synchronizer, Vector<String>& result)
    : DatabaseTask(database, &synchronizer)
    , m_result(result)
{
}

void DatabaseTableNamesTask::doPerformTask()
{
    // FIXME: Why no need for an isolatedCopy here?
    m_result = database().performGetTableNames();
}

#if !LOG_DISABLED

ASCIILiteral DatabaseTableNamesTask::debugTaskName() const
{
    return "DatabaseTableNamesTask"_s;
}

#endif

} // namespace WebCore
