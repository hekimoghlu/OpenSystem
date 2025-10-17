/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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
#include "ChangeVersionWrapper.h"

#include "Database.h"
#include "SQLError.h"
#include "SQLTransactionBackend.h"
#include <wtf/RefPtr.h>

namespace WebCore {

ChangeVersionWrapper::ChangeVersionWrapper(String&& oldVersion, String&& newVersion)
    : m_oldVersion(WTFMove(oldVersion).isolatedCopy())
    , m_newVersion(WTFMove(newVersion).isolatedCopy())
{
}

bool ChangeVersionWrapper::performPreflight(SQLTransaction& transaction)
{
    Database& database = transaction.database();

    String actualVersion;
    if (!database.getVersionFromDatabase(actualVersion)) {
        int sqliteError = database.sqliteDatabase().lastError();
        m_sqlError = SQLError::create(SQLError::UNKNOWN_ERR, "unable to read the current version"_s, sqliteError, database.sqliteDatabase().lastErrorMsg());
        return false;
    }

    if (actualVersion != m_oldVersion) {
        m_sqlError = SQLError::create(SQLError::VERSION_ERR, "current version of the database and `oldVersion` argument do not match"_s);
        return false;
    }

    return true;
}

bool ChangeVersionWrapper::performPostflight(SQLTransaction& transaction)
{
    Database& database = transaction.database();

    if (!database.setVersionInDatabase(m_newVersion)) {
        int sqliteError = database.sqliteDatabase().lastError();
        m_sqlError = SQLError::create(SQLError::UNKNOWN_ERR, "unable to set new version in database"_s, sqliteError, database.sqliteDatabase().lastErrorMsg());
        return false;
    }

    database.setExpectedVersion(m_newVersion);
    return true;
}

void ChangeVersionWrapper::handleCommitFailedAfterPostflight(SQLTransaction& transaction)
{
    transaction.database().setCachedVersion(m_oldVersion);
}

} // namespace WebCore
