/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#include "DatabaseThread.h"

#include "Database.h"
#include "DatabaseTask.h"
#include "Logging.h"
#include "SQLTransaction.h"
#include "SQLTransactionCoordinator.h"
#include <wtf/AutodrainedPool.h>

namespace WebCore {

DatabaseThread::DatabaseThread()
    : m_transactionCoordinator(makeUnique<SQLTransactionCoordinator>())
{
    m_selfRef = this;
}

DatabaseThread::~DatabaseThread()
{
    // The DatabaseThread will only be destructed when both its owner
    // DatabaseContext has deref'ed it, and the databaseThread() thread function
    // has deref'ed the DatabaseThread object. The DatabaseContext destructor
    // will take care of ensuring that a termination request has been issued.
    // The termination request will trigger an orderly shutdown of the thread
    // function databaseThread(). In shutdown, databaseThread() will deref the
    // DatabaseThread before returning.
    ASSERT(terminationRequested());
}

void DatabaseThread::start()
{
    Locker locker { m_threadCreationMutex };

    if (m_thread)
        return;

    m_thread = Thread::create("WebCore: Database"_s, [this] {
        databaseThread();
    });
}

void DatabaseThread::requestTermination(DatabaseTaskSynchronizer* cleanupSync)
{
    m_cleanupSync = cleanupSync;
    LOG(StorageAPI, "DatabaseThread %p was asked to terminate\n", this);
    m_queue.kill();
}

bool DatabaseThread::terminationRequested(DatabaseTaskSynchronizer* taskSynchronizer) const
{
#if ASSERT_ENABLED
    if (taskSynchronizer)
        taskSynchronizer->setHasCheckedForTermination();
#else
    UNUSED_PARAM(taskSynchronizer);
#endif

    return m_queue.killed();
}

void DatabaseThread::databaseThread()
{
    {
        // Wait for DatabaseThread::start() to complete.
        Locker locker { m_threadCreationMutex };
        LOG(StorageAPI, "Started DatabaseThread %p", this);
    }

    while (auto task = m_queue.waitForMessage()) {
        AutodrainedPool pool;

        task->performTask();
    }

    // Clean up the list of all pending transactions on this database thread
    m_transactionCoordinator->shutdown();

    LOG(StorageAPI, "About to detach thread %p and clear the ref to DatabaseThread %p, which currently has %i ref(s)", m_thread.get(), this, refCount());

    // Close the databases that we ran transactions on. This ensures that if any transactions are still open, they are rolled back and we don't leave the database in an
    // inconsistent or locked state.
    DatabaseSet openSetCopy;
    {
        Locker locker { m_openDatabaseSetLock };
        if (m_openDatabaseSet.size() > 0) {
            // As the call to close will modify the original set, we must take a copy to iterate over.
            openSetCopy.swap(m_openDatabaseSet);
        }
    }

    for (auto& openDatabase : openSetCopy)
        openDatabase->performClose();

    // Detach the thread so its resources are no longer of any concern to anyone else
    m_thread->detach();

    DatabaseTaskSynchronizer* cleanupSync = m_cleanupSync;

    // Clear the self refptr, possibly resulting in deletion
    m_selfRef = nullptr;

    if (cleanupSync) // Someone wanted to know when we were done cleaning up.
        cleanupSync->taskCompleted();
}

void DatabaseThread::recordDatabaseOpen(Database& database)
{
    Locker locker { m_openDatabaseSetLock };

    ASSERT(m_thread == &Thread::current());
    ASSERT(!m_openDatabaseSet.contains(&database));
    m_openDatabaseSet.add(&database);
}

void DatabaseThread::recordDatabaseClosed(Database& database)
{
    Locker locker { m_openDatabaseSetLock };

    ASSERT(m_thread == &Thread::current());
    ASSERT(m_queue.killed() || m_openDatabaseSet.contains(&database));
    m_openDatabaseSet.remove(&database);
}

void DatabaseThread::scheduleTask(std::unique_ptr<DatabaseTask>&& task)
{
    ASSERT(!task->hasSynchronizer() || task->hasCheckedForTermination());
    m_queue.append(WTFMove(task));
}

void DatabaseThread::scheduleImmediateTask(std::unique_ptr<DatabaseTask>&& task)
{
    ASSERT(!task->hasSynchronizer() || task->hasCheckedForTermination());
    m_queue.prepend(WTFMove(task));
}

void DatabaseThread::unscheduleDatabaseTasks(Database& database)
{
    // The thread loop is running, sp some tasks for this database may still be executed. This is unavoidable.
    m_queue.removeIf([&database] (const DatabaseTask& task) {
        return &task.database() == &database;
    });
}

bool DatabaseThread::hasPendingDatabaseActivity() const
{
    Locker locker { m_openDatabaseSetLock };
    for (auto& database : m_openDatabaseSet) {
        if (database->hasPendingCreationEvent() || database->hasPendingTransaction())
            return true;
    }
    return false;
}

} // namespace WebCore
