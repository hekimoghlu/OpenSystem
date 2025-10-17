/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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

#include "SQLTransactionStateMachine.h"
#include <memory>
#include <wtf/Forward.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Database;
class OriginLock;
class SQLError;
class SQLiteTransaction;
class SQLStatement;
class SQLTransaction;
class SQLTransactionWrapper;

class SQLTransactionBackend : public SQLTransactionStateMachine<SQLTransactionBackend> {
public:
    explicit SQLTransactionBackend(SQLTransaction&);
    ~SQLTransactionBackend();

    void notifyDatabaseThreadIsShuttingDown();

    // API called from the frontend published via SQLTransactionBackend:
    void requestTransitToState(SQLTransactionState);

private:
    friend class SQLTransaction;

    void doCleanup();

    // State Machine functions:
    StateFunction stateFunctionFor(SQLTransactionState) override;
    void computeNextStateAndCleanupIfNeeded();

    // State functions:
    void acquireLock();
    void openTransactionAndPreflight();
    void runStatements();
    void cleanupAndTerminate();
    void cleanupAfterTransactionErrorCallback();

    NO_RETURN_DUE_TO_ASSERT void unreachableState();

    SQLTransaction& m_frontend;
};

} // namespace WebCore
