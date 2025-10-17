/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 7, 2025.
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
#include "WebExtensionSQLiteDatabase.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#include "APIError.h"
#include "Logging.h"
#include "WebExtensionSQLiteHelpers.h"
#include <sqlite3.h>
#include <wtf/FileSystem.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>

static constexpr auto WebExtensionSQLiteErrorDomain = "com.apple.WebKit.SQLite"_s;
static constexpr auto WebExtensionSQLiteInMemoryDatabaseName = "file::memory:"_s;
static constexpr auto WebExtensionSQLiteErrorMessageKey = "Message"_s;
static constexpr auto WebExtensionSQLiteErrorSQLKey = "SQL"_s;

using namespace WebKit;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebExtensionSQLiteDatabase);

WebExtensionSQLiteDatabase::WebExtensionSQLiteDatabase(const URL& url, Ref<WorkQueue>&& queue)
    : m_queue(WTFMove(queue))
{
    ASSERT(url.protocolIsFile());

    m_url = url;
}

void WebExtensionSQLiteDatabase::assertQueue()
{
    assertIsCurrent(m_queue.get());
}

int WebExtensionSQLiteDatabase::close()
{
    assertQueue();

    int result = sqlite3_close_v2(m_db);
    if (result != SQLITE_OK) {
        RELEASE_LOG_ERROR(Extensions, "Failed to close database: %s (%d)", m_lastErrorMessage.data(), result);
        return result;
    }

    m_db = nullptr;
    return result;
}

void WebExtensionSQLiteDatabase::reportErrorWithCode(int errorCode, const String& query, RefPtr<API::Error>& outError)
{
    assertQueue();
    ASSERT(errorCode != SQLITE_OK);

    if (!query.isEmpty())
        RELEASE_LOG_ERROR(Extensions, "SQLite error (%d) occurred with query: %{private}s", errorCode, query.utf8().data());
    else
        RELEASE_LOG_ERROR(Extensions, "SQLite error (%d) occurred", errorCode);

    outError = errorWithSQLiteErrorCode(errorCode);
}

void WebExtensionSQLiteDatabase::reportErrorWithCode(int errorCode, sqlite3_stmt* statement, RefPtr<API::Error>& outError)
{
    assertQueue();
    ASSERT(errorCode != SQLITE_OK);

    if (statement) {
        if (char* sql = sqlite3_expanded_sql(statement)) {
            reportErrorWithCode(errorCode, String::fromUTF8(sql), outError);
            sqlite3_free(sql);
            return;
        }
    }

    reportErrorWithCode(errorCode, { }, outError);
}

RefPtr<API::Error> WebExtensionSQLiteDatabase::errorWithSQLiteErrorCode(int errorCode)
{
    if (errorCode == SQLITE_OK)
        return nullptr;

    auto errorMessage = String::fromUTF8(sqlite3_errstr(errorCode));
    return API::Error::create({ WebExtensionSQLiteErrorDomain, errorCode, m_url, errorMessage });
}

bool WebExtensionSQLiteDatabase::enableWAL(RefPtr<API::Error>& error)
{
    Ref<WebExtensionSQLiteDatabase> protectedThis(*this);

    // SQLite docs: The synchronous NORMAL setting is a good choice for most applications running in WAL mode.
    if (!SQLiteDatabaseExecuteAndReturnError(*this, error, "PRAGMA synchronous = NORMAL"_s))
        return false;
    return SQLiteDatabaseEnumerate(*this, error, "PRAGMA journal_mode = WAL"_s, std::tie(std::ignore));
}

bool WebExtensionSQLiteDatabase::openWithAccessType(AccessType accessType, RefPtr<API::Error>& outError, ProtectionType protectionType, const String& vfs)
{
    int flags = SQLITE_OPEN_NOMUTEX;

    switch (accessType) {
    case AccessType::ReadOnly:
        flags |= SQLITE_OPEN_READONLY;
        break;
    case AccessType::ReadWrite:
        flags |= SQLITE_OPEN_READWRITE;
        break;
    case AccessType::ReadWriteCreate:
        flags |= SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
        break;
    }

#if PLATFORM(IOS_FAMILY)
    switch (protectionType) {
    case ProtectionType::Default:
    case ProtectionType::CompleteUntilFirstUserAuthentication:
        flags |= SQLITE_OPEN_FILEPROTECTION_COMPLETEUNTILFIRSTUSERAUTHENTICATION;
        break;
    case ProtectionType::CompleteUnlessOpen:
        flags |= SQLITE_OPEN_FILEPROTECTION_COMPLETEUNLESSOPEN;
        break;
    case ProtectionType::Complete:
        flags |= SQLITE_OPEN_FILEPROTECTION_COMPLETE;
        break;
    }
#endif

    assertQueue();
    ASSERT(!m_db);

    String databasePath;
    if (m_url == inMemoryDatabaseURL())
        databasePath = WebExtensionSQLiteInMemoryDatabaseName;
    else if (m_url == privateOnDiskDatabaseURL())
        databasePath = ""_s;
    else {
        ASSERT(!m_url.isEmpty());

        databasePath = m_url.fileSystemPath();

        auto directory = m_url.truncatedForUseAsBase().fileSystemPath();
        if (!FileSystem::makeAllDirectories(directory) || FileSystem::fileType(directory) != FileSystem::FileType::Directory) {
            RELEASE_LOG_ERROR(Extensions, "Unable to create parent folder for database at path: %s", m_url.fileSystemPath().utf8().data());
            outError = errorWithSQLiteErrorCode(SQLITE_CANTOPEN);
            return false;
        }
    }

    int result = sqlite3_open_v2(FileSystem::fileSystemRepresentation(databasePath).data(), &m_db, flags, vfs.isEmpty() ? nullptr : vfs.utf8().data());
    if (result == SQLITE_OK)
        return true;

    // SQLite may return a valid database handle even if an error occurred. sqlite3_close silently
    // ignores calls with a null handle so we can call itÂ here unconditionally.
    sqlite3_close_v2(m_db);
    m_db = nullptr;

    if (result == SQLITE_CANTOPEN && !(flags & SQLITE_OPEN_CREATE))
        return false;

    outError = errorWithSQLiteErrorCode(result);

    return false;
}

URL WebExtensionSQLiteDatabase::inMemoryDatabaseURL()
{
    return URL(WebExtensionSQLiteInMemoryDatabaseName);
}

URL WebExtensionSQLiteDatabase::privateOnDiskDatabaseURL()
{
    return URL("webkit::privateondisk"_s);
}

#endif // ENABLE(WK_WEB_EXTENSIONS)
