/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#include "SQLiteTransaction.h"

#include "Logging.h"
#include "SQLiteDatabase.h"
#include "SQLiteDatabaseTracker.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SQLiteTransaction);

SQLiteTransaction::SQLiteTransaction(SQLiteDatabase& db, bool readOnly)
    : m_db(db)
    , m_inProgress(false)
    , m_readOnly(readOnly)
{
}

SQLiteTransaction::~SQLiteTransaction()
{
    if (m_inProgress)
        rollback();
}

void SQLiteTransaction::begin()
{
    if (!m_inProgress) {
        ASSERT(!m_db->m_transactionInProgress);
        // Call BEGIN IMMEDIATE for a write transaction to acquire
        // a RESERVED lock on the DB file. Otherwise, another write
        // transaction (on another connection) could make changes
        // to the same DB file before this transaction gets to execute
        // any statements. If that happens, this transaction will fail.
        // http://www.sqlite.org/lang_transaction.html
        // http://www.sqlite.org/lockingv3.html#locking
        SQLiteDatabaseTracker::incrementTransactionInProgressCount();
        int result = SQLITE_OK;
        if (m_readOnly)
            result = m_db->execute("BEGIN"_s);
        else
            result = m_db->execute("BEGIN IMMEDIATE"_s);
        if (result == SQLITE_DONE)
            m_inProgress = true;
        else
            RELEASE_LOG_ERROR(SQLDatabase, "SQLiteTransaction::begin: Failed to begin transaction (error %d)", result);
        m_db->m_transactionInProgress = m_inProgress;
        if (!m_inProgress)
            SQLiteDatabaseTracker::decrementTransactionInProgressCount();
    } else
        RELEASE_LOG_ERROR(SQLDatabase, "SQLiteTransaction::begin: Transaction is already in progress");
}

void SQLiteTransaction::commit()
{
    if (m_inProgress) {
        ASSERT(m_db->m_transactionInProgress);
        m_inProgress = !m_db->executeCommand("COMMIT"_s);
        m_db->m_transactionInProgress = m_inProgress;
        if (!m_inProgress)
            SQLiteDatabaseTracker::decrementTransactionInProgressCount();
    }
}

void SQLiteTransaction::rollback()
{
    // We do not use the 'm_inProgress = m_db->executeCommand("ROLLBACK")' construct here,
    // because m_inProgress should always be set to false after a ROLLBACK, and
    // m_db->executeCommand("ROLLBACK") can sometimes harmlessly fail, thus returning
    // a non-zero/true result (http://www.sqlite.org/lang_transaction.html).
    if (m_inProgress) {
        ASSERT(m_db->m_transactionInProgress);
        m_db->executeCommand("ROLLBACK"_s);
        m_inProgress = false;
        m_db->m_transactionInProgress = false;
        SQLiteDatabaseTracker::decrementTransactionInProgressCount();
    }
}

void SQLiteTransaction::stop()
{
    if (m_inProgress) {
        m_inProgress = false;
        m_db->m_transactionInProgress = false;
        SQLiteDatabaseTracker::decrementTransactionInProgressCount();
    }
}

bool SQLiteTransaction::wasRolledBackBySqlite() const
{
    // According to http://www.sqlite.org/c3ref/get_autocommit.html,
    // the auto-commit flag should be off in the middle of a transaction
    return m_inProgress && m_db->isAutoCommitOn();
}

} // namespace WebCore
