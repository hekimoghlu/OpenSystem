/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#include "WebExtensionSQLiteStatement.h"

#include "Logging.h"
#include "WebExtensionSQLiteDatabase.h"
#include "WebExtensionSQLiteHelpers.h"
#include <sqlite3.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebExtensionSQLiteStatement);

WebExtensionSQLiteStatement::WebExtensionSQLiteStatement(Ref<WebExtensionSQLiteDatabase> database, const String& query, RefPtr<API::Error>& outError)
    : m_db(database)
{
    Ref db = m_db;

    ASSERT(db->sqlite3Handle());

    db->assertQueue();

    int result = sqlite3_prepare_v2(db->sqlite3Handle(), query.utf8().data(), -1, &m_handle, 0);
    if (result != SQLITE_OK) {
        db->reportErrorWithCode(result, query, outError);
        return;
    }
}

WebExtensionSQLiteStatement::~WebExtensionSQLiteStatement()
{
    sqlite3_stmt* handle = m_handle;
    if (!handle)
        return;

    database()->queue()->dispatch([database = Ref { database() }, handle = WTFMove(handle)]() mutable {
        // The database might have closed already;
        if (!database->sqlite3Handle())
            return;

        sqlite3_finalize(handle);
    });
}

int WebExtensionSQLiteStatement::execute()
{
    database()->assertQueue();
    ASSERT(isValid());

    int resultCode = sqlite3_step(m_handle);
    if (!SQLiteIsExecutionError(resultCode))
        return resultCode;

    return resultCode;
}

bool WebExtensionSQLiteStatement::execute(RefPtr<API::Error>& outError)
{
    Ref database = m_db;

    database->assertQueue();
    ASSERT(isValid());

    int resultCode = sqlite3_step(m_handle);
    if (!SQLiteIsExecutionError(resultCode))
        return true;

    database->reportErrorWithCode(resultCode, m_handle, outError);
    return false;
}

Ref<WebExtensionSQLiteRowEnumerator> WebExtensionSQLiteStatement::fetch()
{
    m_db->assertQueue();
    ASSERT(isValid());

    Ref<WebExtensionSQLiteStatement> protectedThis(*this);
    return WebExtensionSQLiteRowEnumerator::create(*this);
}

bool WebExtensionSQLiteStatement::fetchWithEnumerationCallback(Function<void(RefPtr<WebExtensionSQLiteRow>, bool)>& callback, RefPtr<API::Error>& outError)
{
    m_db->assertQueue();
    ASSERT(isValid());

    RefPtr<WebExtensionSQLiteRow> row;
    Ref<WebExtensionSQLiteStatement> protectedThis(*this);

    int result = SQLITE_OK;
    bool stop = false;
    while (!stop) {
        result = sqlite3_step(m_handle);
        if (result != SQLITE_ROW)
            break;

        if (!row)
            row = WebExtensionSQLiteRow::create(*this);

        callback(row, stop);
    }

    if (result == SQLITE_DONE)
        return true;

    return false;
}

void WebExtensionSQLiteStatement::reset()
{
    m_db->assertQueue();
    ASSERT(isValid());

    int result = sqlite3_reset(m_handle);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not reset statement: %s (%d)", m_db->m_lastErrorMessage.data(), result);
}

void WebExtensionSQLiteStatement::invalidate()
{
    m_db->assertQueue();
    ASSERT(isValid());

    int result = sqlite3_finalize(m_handle);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not finalize statement: %s (%d)", m_db->m_lastErrorMessage.data(), (int)result);
    m_handle = nullptr;
}

void WebExtensionSQLiteStatement::bind(const String& string, int parameterIndex)
{
    m_db->assertQueue();
    ASSERT(isValid());
    ASSERT_ARG(parameterIndex, parameterIndex > 0);

    int result = sqlite3_bind_text(m_handle, parameterIndex, string.utf8().data(), -1, SQLITE_TRANSIENT);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind string: %s (%d)", m_db->m_lastErrorMessage.data(), (int)result);
}

void WebExtensionSQLiteStatement::bind(const int& n, int parameterIndex)
{
    m_db->assertQueue();
    ASSERT(isValid());
    ASSERT_ARG(parameterIndex, parameterIndex > 0);

    int result = sqlite3_bind_int(m_handle, parameterIndex, n);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind int: %s (%d)", m_db->m_lastErrorMessage.data(), (int)result);
}

void WebExtensionSQLiteStatement::bind(const int64_t& n, int parameterIndex)
{
    m_db->assertQueue();
    ASSERT(isValid());
    ASSERT_ARG(parameterIndex, parameterIndex > 0);

    int result = sqlite3_bind_int64(m_handle, parameterIndex, n);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind integer: %s (%d)", m_db->m_lastErrorMessage.data(), (int)result);
}

void WebExtensionSQLiteStatement::bind(const double& n, int parameterIndex)
{
    m_db->assertQueue();
    ASSERT(isValid());
    ASSERT_ARG(parameterIndex, parameterIndex > 0);

    int result = sqlite3_bind_double(m_handle, parameterIndex, n);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind int: %s (%d)", m_db->m_lastErrorMessage.data(), (int)result);
}

void WebExtensionSQLiteStatement::bind(const RefPtr<API::Data>& data, int parameterIndex)
{
    m_db->assertQueue();
    ASSERT(isValid());
    ASSERT_ARG(parameterIndex, parameterIndex > 0);

    int result = sqlite3_bind_blob(m_handle, parameterIndex, data->span().data(), data->span().size(), SQLITE_TRANSIENT);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind blob: %s (%d)", m_db->m_lastErrorMessage.data(), (int)result);
}

void WebExtensionSQLiteStatement::bind(int parameterIndex)
{
    m_db->assertQueue();
    ASSERT(isValid());
    ASSERT_ARG(parameterIndex, parameterIndex > 0);

    int result = sqlite3_bind_null(m_handle, parameterIndex);
    if (result != SQLITE_OK)
        RELEASE_LOG_DEBUG(Extensions, "Could not bind null: %s (%d)", m_db->m_lastErrorMessage.data(), (int)result);
}

HashMap<String, int> WebExtensionSQLiteStatement::columnNamesToIndicies()
{
    m_db->assertQueue();
    ASSERT(isValid());

    if (!m_columnNamesToIndicies.isEmpty())
        return m_columnNamesToIndicies;

    int columnCount = sqlite3_column_count(m_handle);
    m_columnNamesToIndicies.reserveInitialCapacity(columnCount);

    for (int i = 0; i < columnCount; i++) {
        const char* columnName = sqlite3_column_name(m_handle, i);
        ASSERT(columnName);

        m_columnNamesToIndicies.add(String::fromUTF8(columnName), i);
    }

    return m_columnNamesToIndicies;
}

Vector<String> WebExtensionSQLiteStatement::columnNames()
{
    m_db->assertQueue();
    ASSERT(isValid());

    if (!m_columnNames.isEmpty())
        return m_columnNames;

    int columnCount = sqlite3_column_count(m_handle);
    m_columnNames.reserveInitialCapacity(columnCount);

    for (int i = 0; i < columnCount; i++) {
        const char* columnName = sqlite3_column_name(m_handle, i);
        ASSERT(columnName);

        m_columnNames.append(String::fromUTF8(columnName));
    }

    return m_columnNames;
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
