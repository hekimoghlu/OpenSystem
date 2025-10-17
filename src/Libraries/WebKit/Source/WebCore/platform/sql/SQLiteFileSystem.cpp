/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#include "SQLiteFileSystem.h"

#include "Logging.h"
#include "SQLiteDatabase.h"
#include "SQLiteStatement.h"
#include <pal/crypto/CryptoDigest.h>
#include <sqlite3.h>
#include <wtf/FileSystem.h>
#include <wtf/HexNumber.h>
#include <wtf/text/MakeString.h>

#if PLATFORM(COCOA)
#include <pal/spi/cocoa/SQLite3SPI.h>
#endif

#if PLATFORM(COCOA)
#include <sys/xattr.h>
#endif

namespace WebCore {

static constexpr std::array<ASCIILiteral, 3> databaseFileSuffixes { ""_s, "-shm"_s, "-wal"_s };

SQLiteFileSystem::SQLiteFileSystem() = default;

String SQLiteFileSystem::appendDatabaseFileNameToPath(StringView path, StringView fileName)
{
    return FileSystem::pathByAppendingComponent(path, fileName);
}

bool SQLiteFileSystem::ensureDatabaseDirectoryExists(const String& path)
{
    if (path.isEmpty())
        return false;
    return FileSystem::makeAllDirectories(path);
}

bool SQLiteFileSystem::ensureDatabaseFileExists(const String& fileName, bool checkPathOnly)
{
    if (fileName.isEmpty())
        return false;

    if (checkPathOnly) {
        String dir = FileSystem::parentPath(fileName);
        return ensureDatabaseDirectoryExists(dir);
    }

    return FileSystem::fileExists(fileName);
}

bool SQLiteFileSystem::deleteEmptyDatabaseDirectory(const String& path)
{
    return FileSystem::deleteEmptyDirectory(path);
}

bool SQLiteFileSystem::deleteDatabaseFile(const String& filePath)
{
    bool fileExists = false;
    for (auto suffix : databaseFileSuffixes) {
        auto path = makeString(filePath, suffix);
        FileSystem::deleteFile(path);
        fileExists |= FileSystem::fileExists(path);
    }

    return !fileExists;
}

#if PLATFORM(COCOA)
void SQLiteFileSystem::setCanSuspendLockedFileAttribute(const String& filePath)
{
    for (auto suffix : databaseFileSuffixes) {
        auto path = makeString(filePath, suffix);
        char excluded = 0xff;
        auto result = setxattr(FileSystem::fileSystemRepresentation(path).data(), "com.apple.runningboard.can-suspend-locked", &excluded, sizeof(excluded), 0, 0);
        if (result < 0 && suffix == ""_s)
            RELEASE_LOG_ERROR(SQLDatabase, "SQLiteFileSystem::setCanSuspendLockedFileAttribute: setxattr failed: %" PUBLIC_LOG_STRING, strerror(errno));
    }
}
#endif

bool SQLiteFileSystem::moveDatabaseFile(const String& oldFilePath, const String& newFilePath)
{
    bool allMoved = true;
    for (auto suffix : databaseFileSuffixes)
        allMoved &= FileSystem::moveFile(makeString(oldFilePath, suffix), makeString(newFilePath, suffix));

    return allMoved;
}

#if PLATFORM(COCOA)
bool SQLiteFileSystem::truncateDatabaseFile(sqlite3* database)
{
    return sqlite3_file_control(database, 0, SQLITE_TRUNCATE_DATABASE, 0) == SQLITE_OK;
}
#endif
    
uint64_t SQLiteFileSystem::databaseFileSize(const String& filePath)
{
    uint64_t totalSize = 0;
    for (auto suffix : databaseFileSuffixes) {
        if (auto fileSize = FileSystem::fileSize(makeString(filePath, suffix)))
            totalSize += *fileSize;
    }

    return totalSize;
}

std::optional<WallTime> SQLiteFileSystem::databaseCreationTime(const String& fileName)
{
    return FileSystem::fileCreationTime(fileName);
}

std::optional<WallTime> SQLiteFileSystem::databaseModificationTime(const String& fileName)
{
    return FileSystem::fileModificationTime(fileName);
}
    
String SQLiteFileSystem::computeHashForFileName(StringView fileName)
{
    auto cryptoDigest = PAL::CryptoDigest::create(PAL::CryptoDigest::Algorithm::SHA_256);
    auto utf8FileName = fileName.utf8();
    cryptoDigest->addBytes(utf8FileName.span());
    return cryptoDigest->toHexString();
}

} // namespace WebCore
