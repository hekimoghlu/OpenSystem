/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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
#include "WebExtensionSQLiteRow.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#include "WebExtensionSQLiteDatabase.h"
#include "WebExtensionSQLiteStatement.h"
#include <sqlite3.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebExtensionSQLiteRow);
WTF_MAKE_TZONE_ALLOCATED_IMPL(WebExtensionSQLiteRowEnumerator);

WebExtensionSQLiteRow::WebExtensionSQLiteRow(Ref<WebExtensionSQLiteStatement> statement)
    : m_statement(statement)
    , m_handle(statement->handle())
{
    m_statement->database()->assertQueue();
}

String WebExtensionSQLiteRow::getString(int index)
{
    m_statement->database()->assertQueue();
    if (isNullAtIndex(index))
        return emptyString();

    sqlite3_stmt* handle = m_handle;
    return String::fromUTF8(reinterpret_cast<const char*>(sqlite3_column_text(handle, index)));
}

int WebExtensionSQLiteRow::getInt(int index)
{
    m_statement->database()->assertQueue();
    return sqlite3_column_int(m_handle, index);
}

int64_t WebExtensionSQLiteRow::getInt64(int index)
{
    m_statement->database()->assertQueue();
    return sqlite3_column_int64(m_handle, index);
}

double WebExtensionSQLiteRow::getDouble(int index)
{
    m_statement->database()->assertQueue();
    return sqlite3_column_double(m_handle, index);
}

bool WebExtensionSQLiteRow::getBool(int index)
{
    return !!getInt(index);
}

RefPtr<API::Data> WebExtensionSQLiteRow::getData(int index)
{
    m_statement->database()->assertQueue();
    if (isNullAtIndex(index))
        return nullptr;

    auto* blob = static_cast<const uint8_t*>(sqlite3_column_blob(m_handle, index));
    if (!blob)
        return nullptr;

    int blobSize = sqlite3_column_bytes(m_handle, index);
    if (blobSize <= 0)
        return nullptr;

    return API::Data::create(unsafeMakeSpan(blob, blobSize));
}

bool WebExtensionSQLiteRow::isNullAtIndex(int index)
{
    m_statement->database()->assertQueue();
    return sqlite3_column_type(m_handle, index) == SQLITE_NULL;
}

WebExtensionSQLiteRowEnumerator::WebExtensionSQLiteRowEnumerator(Ref<WebExtensionSQLiteStatement> statement)
    : m_statement(statement)
{
    m_statement->database()->assertQueue();
}

RefPtr<WebExtensionSQLiteRow> WebExtensionSQLiteRowEnumerator::next()
{
    m_statement->database()->assertQueue();

    switch (sqlite3_step(m_statement->handle())) {
    case SQLITE_ROW:
        if (!m_row)
            m_row = WebExtensionSQLiteRow::create(m_statement);
        return m_row;

    default:
        return nullptr;
    }
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
