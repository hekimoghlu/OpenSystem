/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#include "NetworkCacheFileSystem.h"

#include "Logging.h"
#include <wtf/Assertions.h>
#include <wtf/FileSystem.h>
#include <wtf/Function.h>
#include <wtf/text/CString.h>

#if !OS(WINDOWS)
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#else
#include <windows.h>
#endif

#if PLATFORM(IOS_FAMILY) && !PLATFORM(IOS_FAMILY_SIMULATOR)
#include <sys/attr.h>
#include <unistd.h>
#endif

#if USE(SOUP)
#include <gio/gio.h>
#include <wtf/glib/GRefPtr.h>
#endif

namespace WebKit {
namespace NetworkCache {

void traverseDirectory(const String& path, const Function<void (const String&, DirectoryEntryType)>& function)
{
    auto entries = FileSystem::listDirectory(path);
    for (auto& entry : entries) {
        auto entryPath = FileSystem::pathByAppendingComponent(path, entry);
        auto type = FileSystem::fileType(entryPath) == FileSystem::FileType::Directory ? DirectoryEntryType::Directory : DirectoryEntryType::File;
        function(entry, type);
    }
}

FileTimes fileTimes(const String& path)
{
#if HAVE(STAT_BIRTHTIME)
    struct stat fileInfo;
    if (stat(FileSystem::fileSystemRepresentation(path).data(), &fileInfo))
        return { };
    return { WallTime::fromRawSeconds(fileInfo.st_birthtime), WallTime::fromRawSeconds(fileInfo.st_mtime) };
#elif USE(SOUP)
    // There's no st_birthtime in some operating systems like Linux, so we use xattrs to set/get the creation time.
    GRefPtr<GFile> file = adoptGRef(g_file_new_for_path(FileSystem::fileSystemRepresentation(path).data()));
    GRefPtr<GFileInfo> fileInfo = adoptGRef(g_file_query_info(file.get(), "xattr::birthtime,time::modified", G_FILE_QUERY_INFO_NONE, nullptr, nullptr));
    if (!fileInfo)
        return { };
    const char* birthtimeString = g_file_info_get_attribute_string(fileInfo.get(), "xattr::birthtime");
    if (!birthtimeString)
        return { };
    return { WallTime::fromRawSeconds(g_ascii_strtoull(birthtimeString, nullptr, 10)),
        WallTime::fromRawSeconds(g_file_info_get_attribute_uint64(fileInfo.get(), "time::modified")) };
#elif OS(WINDOWS)
    auto createTime = FileSystem::fileCreationTime(path);
    auto modifyTime = FileSystem::fileModificationTime(path);
    return { valueOrDefault(createTime), valueOrDefault(modifyTime) };
#endif
}

void updateFileModificationTimeIfNeeded(const String& path)
{
    auto times = fileTimes(path);
    if (times.creation != times.modification) {
        // Don't update more than once per hour.
        if (WallTime::now() - times.modification < 1_h)
            return;
    }
    FileSystem::updateFileModificationTime(path);
}

}
}
