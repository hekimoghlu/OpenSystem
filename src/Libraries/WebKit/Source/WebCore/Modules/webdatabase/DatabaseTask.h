/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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

#include "ExceptionOr.h"
#include <wtf/Condition.h>
#include <wtf/Forward.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Database;
class SQLTransaction;

// Can be used to wait until DatabaseTask is completed.
// Has to be passed into DatabaseTask::create to be associated with the task.
class DatabaseTaskSynchronizer {
    WTF_MAKE_NONCOPYABLE(DatabaseTaskSynchronizer);
public:
    DatabaseTaskSynchronizer();

    // Called from main thread to wait until task is completed.
    void waitForTaskCompletion();

    // Called by the task.
    void taskCompleted();

#if ASSERT_ENABLED
    bool hasCheckedForTermination() const { return m_hasCheckedForTermination; }
    void setHasCheckedForTermination() { m_hasCheckedForTermination = true; }
#endif

private:
    bool m_taskCompleted WTF_GUARDED_BY_LOCK(m_synchronousLock) { false };
    Lock m_synchronousLock;
    Condition m_synchronousCondition;
#if ASSERT_ENABLED
    bool m_hasCheckedForTermination { false };
#endif
};

class DatabaseTask {
    WTF_MAKE_TZONE_ALLOCATED(DatabaseTask);
public:
    virtual ~DatabaseTask();

    void performTask();

    Database& database() const { return m_database; }

#if ASSERT_ENABLED
    bool hasSynchronizer() const { return m_synchronizer; }
    bool hasCheckedForTermination() const { return m_synchronizer->hasCheckedForTermination(); }
#endif

protected:
    DatabaseTask(Database&, DatabaseTaskSynchronizer*);

private:
    virtual void doPerformTask() = 0;

    Database& m_database;
    DatabaseTaskSynchronizer* m_synchronizer;

#if !LOG_DISABLED
    virtual ASCIILiteral debugTaskName() const = 0;
#endif

#if ASSERT_ENABLED
    bool m_complete { false };
#endif
};

class DatabaseOpenTask final : public DatabaseTask {
public:
    DatabaseOpenTask(Database&, bool setVersionInNewDatabase, DatabaseTaskSynchronizer&, ExceptionOr<void>& result);

private:
    void doPerformTask() final;

#if !LOG_DISABLED
    ASCIILiteral debugTaskName() const final;
#endif

    bool m_setVersionInNewDatabase;
    ExceptionOr<void>& m_result;
};

class DatabaseCloseTask final : public DatabaseTask {
public:
    DatabaseCloseTask(Database&, DatabaseTaskSynchronizer&);

private:
    void doPerformTask() final;

#if !LOG_DISABLED
    ASCIILiteral debugTaskName() const final;
#endif
};

class DatabaseTransactionTask final : public DatabaseTask {
public:
    explicit DatabaseTransactionTask(RefPtr<SQLTransaction>&&);
    virtual ~DatabaseTransactionTask();

    SQLTransaction* transaction() const { return m_transaction.get(); }

private:
    void doPerformTask() final;

#if !LOG_DISABLED
    ASCIILiteral debugTaskName() const final;
#endif

    RefPtr<SQLTransaction> m_transaction;
    bool m_didPerformTask;
};

class DatabaseTableNamesTask final : public DatabaseTask {
public:
    DatabaseTableNamesTask(Database&, DatabaseTaskSynchronizer&, Vector<String>& result);

private:
    void doPerformTask() final;

#if !LOG_DISABLED
    ASCIILiteral debugTaskName() const override;
#endif

    Vector<String>& m_result;
};

} // namespace WebCore
