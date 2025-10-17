/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include "SQLCallbackWrapper.h"
#include "SQLTransactionBackend.h"
#include "SQLTransactionStateMachine.h"
#include "SQLValue.h"
#include <wtf/Deque.h>
#include <wtf/Lock.h>

namespace WebCore {

class Database;
class SQLError;
class SQLStatementCallback;
class SQLStatementErrorCallback;
class SQLTransactionBackend;
class SQLTransactionCallback;
class SQLTransactionErrorCallback;
class VoidCallback;

class SQLTransactionWrapper : public ThreadSafeRefCounted<SQLTransactionWrapper> {
public:
    virtual ~SQLTransactionWrapper() = default;
    virtual bool performPreflight(SQLTransaction&) = 0;
    virtual bool performPostflight(SQLTransaction&) = 0;
    virtual SQLError* sqlError() const = 0;
    virtual void handleCommitFailedAfterPostflight(SQLTransaction&) = 0;
};

class SQLTransaction : public ThreadSafeRefCounted<SQLTransaction>, public SQLTransactionStateMachine<SQLTransaction> {
public:
    static Ref<SQLTransaction> create(Ref<Database>&&, RefPtr<SQLTransactionCallback>&&, RefPtr<VoidCallback>&& successCallback, RefPtr<SQLTransactionErrorCallback>&&, RefPtr<SQLTransactionWrapper>&&, bool readOnly);
    ~SQLTransaction();

    ExceptionOr<void> executeSql(const String& sqlStatement, std::optional<Vector<SQLValue>>&& arguments, RefPtr<SQLStatementCallback>&&, RefPtr<SQLStatementErrorCallback>&&);

    void lockAcquired();
    void performNextStep();
    void performPendingCallback();

    Database& database() { return m_database; }
    bool isReadOnly() const { return m_readOnly; }
    void notifyDatabaseThreadIsShuttingDown();

    // APIs called from the backend published via SQLTransaction:
    void requestTransitToState(SQLTransactionState);

private:
    friend class SQLTransactionBackend;

    SQLTransaction(Ref<Database>&&, RefPtr<SQLTransactionCallback>&&, RefPtr<VoidCallback>&& successCallback, RefPtr<SQLTransactionErrorCallback>&&, RefPtr<SQLTransactionWrapper>&&, bool readOnly);

    void enqueueStatement(std::unique_ptr<SQLStatement>);

    void checkAndHandleClosedDatabase();

    void clearCallbackWrappers();

    void scheduleCallback(void (SQLTransaction::*)());

    // State Machine functions:
    StateFunction stateFunctionFor(SQLTransactionState) override;
    void computeNextStateAndCleanupIfNeeded();

    // State functions:
    void acquireLock();
    void openTransactionAndPreflight();
    void runStatements();
    void cleanupAndTerminate();
    void cleanupAfterTransactionErrorCallback();
    void deliverTransactionCallback();
    void deliverTransactionErrorCallback();
    void deliverStatementCallback();
    void deliverQuotaIncreaseCallback();
    void deliverSuccessCallback();

    NO_RETURN_DUE_TO_ASSERT void unreachableState();

    void callErrorCallbackDueToInterruption();

    void getNextStatement();
    bool runCurrentStatement();
    void handleCurrentStatementError();
    void handleTransactionError();
    void postflightAndCommit();

    void acquireOriginLock();
    void releaseOriginLockIfNeeded();

#if !LOG_DISABLED
    static ASCIILiteral debugStepName(void (SQLTransaction::*)());
#endif

    Ref<Database> m_database;
    SQLCallbackWrapper<SQLTransactionCallback> m_callbackWrapper;
    SQLCallbackWrapper<VoidCallback> m_successCallbackWrapper;
    SQLCallbackWrapper<SQLTransactionErrorCallback> m_errorCallbackWrapper;

    RefPtr<SQLTransactionWrapper> m_wrapper;

    void (SQLTransaction::*m_nextStep)();

    bool m_executeSqlAllowed { false };
    RefPtr<SQLError> m_transactionError;

    bool m_shouldRetryCurrentStatement { false };
    bool m_modifiedDatabase { false };
    bool m_lockAcquired { false };
    bool m_readOnly { false };
    bool m_hasVersionMismatch { false };

    Lock m_statementLock;
    Deque<std::unique_ptr<SQLStatement>> m_statementQueue WTF_GUARDED_BY_LOCK(m_statementLock);

    std::unique_ptr<SQLStatement> m_currentStatement;

    std::unique_ptr<SQLiteTransaction> m_sqliteTransaction;
    RefPtr<OriginLock> m_originLock;

    SQLTransactionBackend m_backend;
};

} // namespace WebCore
