/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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
#include "SQLiteDatabaseTracker.h"

#include <mutex>
#include <wtf/Lock.h>

namespace WebCore {

namespace SQLiteDatabaseTracker {

static Lock transactionInProgressLock;
static SQLiteDatabaseTrackerClient* s_staticSQLiteDatabaseTrackerClient WTF_GUARDED_BY_LOCK(transactionInProgressLock) { nullptr };
static unsigned s_transactionInProgressCounter WTF_GUARDED_BY_LOCK(transactionInProgressLock) { 0 };

void setClient(SQLiteDatabaseTrackerClient* client)
{
    Locker locker { transactionInProgressLock };
    s_staticSQLiteDatabaseTrackerClient = client;
}

void incrementTransactionInProgressCount()
{
    Locker locker { transactionInProgressLock };
    if (!s_staticSQLiteDatabaseTrackerClient)
        return;

    s_transactionInProgressCounter++;
    if (s_transactionInProgressCounter == 1)
        s_staticSQLiteDatabaseTrackerClient->willBeginFirstTransaction();
}

void decrementTransactionInProgressCount()
{
    Locker locker { transactionInProgressLock };
    if (!s_staticSQLiteDatabaseTrackerClient)
        return;

    ASSERT(s_transactionInProgressCounter);
    s_transactionInProgressCounter--;

    if (!s_transactionInProgressCounter)
        s_staticSQLiteDatabaseTrackerClient->didFinishLastTransaction();
}

bool hasTransactionInProgress()
{
    Locker locker { transactionInProgressLock };
    return !s_staticSQLiteDatabaseTrackerClient || s_transactionInProgressCounter > 0;
}

} // namespace SQLiteDatabaseTracker

} // namespace WebCore
