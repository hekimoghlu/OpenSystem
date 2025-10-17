/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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
#include "DatabaseAuthorizer.h"

#include <wtf/text/WTFString.h>

namespace WebCore {

Ref<DatabaseAuthorizer> DatabaseAuthorizer::create(const String& databaseInfoTableName)
{
    return adoptRef(*new DatabaseAuthorizer(databaseInfoTableName));
}

DatabaseAuthorizer::DatabaseAuthorizer(const String& databaseInfoTableName)
    : m_securityEnabled(false)
    , m_databaseInfoTableName(databaseInfoTableName)
{
    reset();
    addAllowedFunctions();
}

void DatabaseAuthorizer::reset()
{
    m_lastActionWasInsert = false;
    m_lastActionChangedDatabase = false;
    m_permissions = ReadWriteMask;
}

void DatabaseAuthorizer::resetDeletes()
{
    m_hadDeletes = false;
}

void DatabaseAuthorizer::addAllowedFunctions()
{
    // SQLite functions used to help implement some operations
    // ALTER TABLE helpers
    m_allowedFunctions.add("sqlite_rename_table"_s);
    m_allowedFunctions.add("sqlite_rename_trigger"_s);
    // GLOB helpers
    m_allowedFunctions.add("glob"_s);

    // SQLite core functions
    m_allowedFunctions.add("abs"_s);
    m_allowedFunctions.add("changes"_s);
    m_allowedFunctions.add("coalesce"_s);
    m_allowedFunctions.add("glob"_s);
    m_allowedFunctions.add("ifnull"_s);
    m_allowedFunctions.add("hex"_s);
    m_allowedFunctions.add("last_insert_rowid"_s);
    m_allowedFunctions.add("length"_s);
    m_allowedFunctions.add("like"_s);
    m_allowedFunctions.add("lower"_s);
    m_allowedFunctions.add("ltrim"_s);
    m_allowedFunctions.add("max"_s);
    m_allowedFunctions.add("min"_s);
    m_allowedFunctions.add("nullif"_s);
    m_allowedFunctions.add("quote"_s);
    m_allowedFunctions.add("replace"_s);
    m_allowedFunctions.add("round"_s);
    m_allowedFunctions.add("rtrim"_s);
    m_allowedFunctions.add("soundex"_s);
    m_allowedFunctions.add("sqlite_source_id"_s);
    m_allowedFunctions.add("sqlite_version"_s);
    m_allowedFunctions.add("substr"_s);
    m_allowedFunctions.add("total_changes"_s);
    m_allowedFunctions.add("trim"_s);
    m_allowedFunctions.add("typeof"_s);
    m_allowedFunctions.add("upper"_s);
    m_allowedFunctions.add("zeroblob"_s);

    // SQLite date and time functions
    m_allowedFunctions.add("date"_s);
    m_allowedFunctions.add("time"_s);
    m_allowedFunctions.add("datetime"_s);
    m_allowedFunctions.add("julianday"_s);
    m_allowedFunctions.add("strftime"_s);

    // SQLite aggregate functions
    // max() and min() are already in the list
    m_allowedFunctions.add("avg"_s);
    m_allowedFunctions.add("count"_s);
    m_allowedFunctions.add("group_concat"_s);
    m_allowedFunctions.add("sum"_s);
    m_allowedFunctions.add("total"_s);

    // SQLite FTS functions
    m_allowedFunctions.add("match"_s);
    m_allowedFunctions.add("snippet"_s);
    m_allowedFunctions.add("offsets"_s);
    m_allowedFunctions.add("optimize"_s);

    // SQLite ICU functions
    // like(), lower() and upper() are already in the list
    m_allowedFunctions.add("regexp"_s);
}

int DatabaseAuthorizer::createTable(const String& tableName)
{
    if (!allowWrite())
        return SQLAuthDeny;

    m_lastActionChangedDatabase = true;
    return denyBasedOnTableName(tableName);
}

int DatabaseAuthorizer::createTempTable(const String& tableName)
{
    // SQLITE_CREATE_TEMP_TABLE results in a UPDATE operation, which is not
    // allowed in read-only transactions or private browsing, so we might as
    // well disallow SQLITE_CREATE_TEMP_TABLE in these cases
    if (!allowWrite())
        return SQLAuthDeny;

    return denyBasedOnTableName(tableName);
}

int DatabaseAuthorizer::dropTable(const String& tableName)
{
    if (!allowWrite())
        return SQLAuthDeny;

    return updateDeletesBasedOnTableName(tableName);
}

int DatabaseAuthorizer::dropTempTable(const String& tableName)
{
    // SQLITE_DROP_TEMP_TABLE results in a DELETE operation, which is not
    // allowed in read-only transactions or private browsing, so we might as
    // well disallow SQLITE_DROP_TEMP_TABLE in these cases
    if (!allowWrite())
        return SQLAuthDeny;

    return updateDeletesBasedOnTableName(tableName);
}

int DatabaseAuthorizer::allowAlterTable(const String&, const String& tableName)
{
    if (!allowWrite())
        return SQLAuthDeny;

    m_lastActionChangedDatabase = true;
    return denyBasedOnTableName(tableName);
}

int DatabaseAuthorizer::createIndex(const String&, const String& tableName)
{
    if (!allowWrite())
        return SQLAuthDeny;

    m_lastActionChangedDatabase = true;
    return denyBasedOnTableName(tableName);
}

int DatabaseAuthorizer::createTempIndex(const String&, const String& tableName)
{
    // SQLITE_CREATE_TEMP_INDEX should result in a UPDATE or INSERT operation,
    // which is not allowed in read-only transactions or private browsing,
    // so we might as well disallow SQLITE_CREATE_TEMP_INDEX in these cases
    if (!allowWrite())
        return SQLAuthDeny;

    return denyBasedOnTableName(tableName);
}

int DatabaseAuthorizer::dropIndex(const String&, const String& tableName)
{
    if (!allowWrite())
        return SQLAuthDeny;

    return updateDeletesBasedOnTableName(tableName);
}

int DatabaseAuthorizer::dropTempIndex(const String&, const String& tableName)
{
    // SQLITE_DROP_TEMP_INDEX should result in a DELETE operation, which is
    // not allowed in read-only transactions or private browsing, so we might
    // as well disallow SQLITE_DROP_TEMP_INDEX in these cases
    if (!allowWrite())
        return SQLAuthDeny;

    return updateDeletesBasedOnTableName(tableName);
}

int DatabaseAuthorizer::createTrigger(const String&, const String& tableName)
{
    if (!allowWrite())
        return SQLAuthDeny;

    m_lastActionChangedDatabase = true;
    return denyBasedOnTableName(tableName);
}

int DatabaseAuthorizer::createTempTrigger(const String&, const String& tableName)
{
    // SQLITE_CREATE_TEMP_TRIGGER results in a INSERT operation, which is not
    // allowed in read-only transactions or private browsing, so we might as
    // well disallow SQLITE_CREATE_TEMP_TRIGGER in these cases
    if (!allowWrite())
        return SQLAuthDeny;

    return denyBasedOnTableName(tableName);
}

int DatabaseAuthorizer::dropTrigger(const String&, const String& tableName)
{
    if (!allowWrite())
        return SQLAuthDeny;

    return updateDeletesBasedOnTableName(tableName);
}

int DatabaseAuthorizer::dropTempTrigger(const String&, const String& tableName)
{
    // SQLITE_DROP_TEMP_TRIGGER results in a DELETE operation, which is not
    // allowed in read-only transactions or private browsing, so we might as
    // well disallow SQLITE_DROP_TEMP_TRIGGER in these cases
    if (!allowWrite())
        return SQLAuthDeny;

    return updateDeletesBasedOnTableName(tableName);
}

int DatabaseAuthorizer::createView(const String&)
{
    return (!allowWrite() ? SQLAuthDeny : SQLAuthAllow);
}

int DatabaseAuthorizer::createTempView(const String&)
{
    // SQLITE_CREATE_TEMP_VIEW results in a UPDATE operation, which is not
    // allowed in read-only transactions or private browsing, so we might as
    // well disallow SQLITE_CREATE_TEMP_VIEW in these cases
    return (!allowWrite() ? SQLAuthDeny : SQLAuthAllow);
}

int DatabaseAuthorizer::dropView(const String&)
{
    if (!allowWrite())
        return SQLAuthDeny;

    m_hadDeletes = true;
    return SQLAuthAllow;
}

int DatabaseAuthorizer::dropTempView(const String&)
{
    // SQLITE_DROP_TEMP_VIEW results in a DELETE operation, which is not
    // allowed in read-only transactions or private browsing, so we might as
    // well disallow SQLITE_DROP_TEMP_VIEW in these cases
    if (!allowWrite())
        return SQLAuthDeny;

    m_hadDeletes = true;
    return SQLAuthAllow;
}

int DatabaseAuthorizer::createVTable(const String&, const String&)
{
    return SQLAuthDeny;
}

int DatabaseAuthorizer::dropVTable(const String&, const String&)
{
    return SQLAuthDeny;
}

int DatabaseAuthorizer::allowDelete(const String& tableName)
{
    if (!allowWrite())
        return SQLAuthDeny;

    return updateDeletesBasedOnTableName(tableName);
}

int DatabaseAuthorizer::allowInsert(const String& tableName)
{
    if (!allowWrite())
        return SQLAuthDeny;

    m_lastActionChangedDatabase = true;
    m_lastActionWasInsert = true;
    return denyBasedOnTableName(tableName);
}

int DatabaseAuthorizer::allowUpdate(const String& tableName, const String&)
{
    if (!allowWrite())
        return SQLAuthDeny;

    m_lastActionChangedDatabase = true;
    return denyBasedOnTableName(tableName);
}

int DatabaseAuthorizer::allowTransaction()
{
    return m_securityEnabled ? SQLAuthDeny : SQLAuthAllow;
}

int DatabaseAuthorizer::allowRead(const String& tableName, const String&)
{
    if (m_permissions & NoAccessMask && m_securityEnabled)
        return SQLAuthDeny;

    return denyBasedOnTableName(tableName);
}

int DatabaseAuthorizer::allowReindex(const String&)
{
    return (!allowWrite() ? SQLAuthDeny : SQLAuthAllow);
}

int DatabaseAuthorizer::allowAnalyze(const String& tableName)
{
    return denyBasedOnTableName(tableName);
}

int DatabaseAuthorizer::allowPragma(const String&, const String&)
{
    return m_securityEnabled ? SQLAuthDeny : SQLAuthAllow;
}

int DatabaseAuthorizer::allowAttach(const String&)
{
    return m_securityEnabled ? SQLAuthDeny : SQLAuthAllow;
}

int DatabaseAuthorizer::allowDetach(const String&)
{
    return m_securityEnabled ? SQLAuthDeny : SQLAuthAllow;
}

int DatabaseAuthorizer::allowFunction(const String& functionName)
{
    if (m_securityEnabled && !m_allowedFunctions.contains(functionName))
        return SQLAuthDeny;

    return SQLAuthAllow;
}

void DatabaseAuthorizer::disable()
{
    m_securityEnabled = false;
}

void DatabaseAuthorizer::enable()
{
    m_securityEnabled = true;
}

bool DatabaseAuthorizer::allowWrite()
{
    return !(m_securityEnabled && (m_permissions & ReadOnlyMask || m_permissions & NoAccessMask));
}

void DatabaseAuthorizer::setPermissions(int permissions)
{
    m_permissions = permissions;
}

int DatabaseAuthorizer::denyBasedOnTableName(const String& tableName) const
{
    if (!m_securityEnabled)
        return SQLAuthAllow;

    // Sadly, normal creates and drops end up affecting sqlite_master in an authorizer callback, so
    // it will be tough to enforce all of the following policies.
    // if (equalIgnoringASCIICase(tableName, "sqlite_master")
    //      || equalIgnoringASCIICase(tableName, "sqlite_temp_master")
    //      || equalIgnoringASCIICase(tableName, "sqlite_sequence")
    //      || equalIgnoringASCIICase(tableName, Database::databaseInfoTableName()))
    //    return SQLAuthDeny;

    if (equalIgnoringASCIICase(tableName, m_databaseInfoTableName))
        return SQLAuthDeny;

    return SQLAuthAllow;
}

int DatabaseAuthorizer::updateDeletesBasedOnTableName(const String& tableName)
{
    int allow = denyBasedOnTableName(tableName);
    if (allow)
        m_hadDeletes = true;
    return allow;
}

} // namespace WebCore
