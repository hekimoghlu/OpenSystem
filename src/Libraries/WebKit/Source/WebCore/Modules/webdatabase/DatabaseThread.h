/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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

#include <memory>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/MessageQueue.h>
#include <wtf/RefPtr.h>
#include <wtf/Threading.h>

namespace WebCore {

class Database;
class DatabaseTask;
class DatabaseTaskSynchronizer;
class Document;
class SQLTransactionCoordinator;

class DatabaseThread : public ThreadSafeRefCounted<DatabaseThread> {
public:
    static Ref<DatabaseThread> create() { return adoptRef(*new DatabaseThread); }
    ~DatabaseThread();

    void start();
    void requestTermination(DatabaseTaskSynchronizer* cleanupSync);
    bool terminationRequested(DatabaseTaskSynchronizer* = nullptr) const;

    void scheduleTask(std::unique_ptr<DatabaseTask>&&);
    void scheduleImmediateTask(std::unique_ptr<DatabaseTask>&&); // This just adds the task to the front of the queue - the caller needs to be extremely careful not to create deadlocks when waiting for completion.
    void unscheduleDatabaseTasks(Database&);
    bool hasPendingDatabaseActivity() const;

    void recordDatabaseOpen(Database&);
    void recordDatabaseClosed(Database&);
    Thread* getThread() { return m_thread.get(); }

    SQLTransactionCoordinator* transactionCoordinator() { return m_transactionCoordinator.get(); }

private:
    DatabaseThread();

    void databaseThread();

    Lock m_threadCreationMutex;
    RefPtr<Thread> m_thread;
    RefPtr<DatabaseThread> m_selfRef;

    MessageQueue<DatabaseTask> m_queue;

    // This set keeps track of the open databases that have been used on this thread.
    using DatabaseSet = HashSet<RefPtr<Database>>;
    mutable Lock m_openDatabaseSetLock;
    DatabaseSet m_openDatabaseSet WTF_GUARDED_BY_LOCK(m_openDatabaseSetLock);

    std::unique_ptr<SQLTransactionCoordinator> m_transactionCoordinator;
    DatabaseTaskSynchronizer* m_cleanupSync { nullptr };
};

} // namespace WebCore
