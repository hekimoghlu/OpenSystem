/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
#include <wtf/FileSystem.h>

#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

#if !OS(WINDOWS)
#include <limits.h>
#include <unistd.h>
#endif

namespace WTF {

namespace FileSystemImpl {

bool validRepresentation(const CString& representation)
{
    auto* data = representation.data();
    return !!data && data[0] != '\0';
}

// Converts a string to something suitable to be displayed to the user.
String filenameForDisplay(const String& string)
{
#if OS(WINDOWS)
    return string;
#else
    auto filename = fileSystemRepresentation(string);
    if (!validRepresentation(filename))
        return string;

    GUniquePtr<gchar> display(g_filename_display_name(filename.data()));
    if (!display)
        return string;
    return String::fromUTF8(display.get());
#endif
}

#if OS(LINUX)
CString currentExecutablePath()
{
    static char readLinkBuffer[PATH_MAX];
    ssize_t result = readlink("/proc/self/exe", readLinkBuffer, PATH_MAX);
    if (result == -1)
        return { };
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // Linux port
    return CString({ readLinkBuffer, static_cast<size_t>(result) });
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
}
#elif OS(HURD)
CString currentExecutablePath()
{
    return { };
}
#elif OS(UNIX)
CString currentExecutablePath()
{
    static char readLinkBuffer[PATH_MAX];
    ssize_t result = readlink("/proc/curproc/file", readLinkBuffer, PATH_MAX);
    if (result == -1)
        return { };
    return CString(readLinkBuffer, result);
}
#elif OS(WINDOWS)
CString currentExecutablePath()
{
    static WCHAR buffer[MAX_PATH];
    DWORD length = GetModuleFileNameW(0, buffer, MAX_PATH);
    if (!length || (length == MAX_PATH && GetLastError() == ERROR_INSUFFICIENT_BUFFER))
        return { };

    String path(buffer, length);
    return path.utf8();
}
#endif

CString currentExecutableName()
{
    auto executablePath = currentExecutablePath();
    if (!executablePath.isNull()) {
        GUniquePtr<char> basename(g_path_get_basename(executablePath.data()));
        return basename.get();
    }

    return g_get_prgname();
}

String userCacheDirectory()
{
    return stringFromFileSystemRepresentation(g_get_user_cache_dir());
}

String userDataDirectory()
{
    return stringFromFileSystemRepresentation(g_get_user_data_dir());
}

#if ENABLE(DEVELOPER_MODE)
CString webkitTopLevelDirectory()
{
    if (const char* topLevelDirectory = g_getenv("WEBKIT_TOP_LEVEL")) {
        if (g_file_test(topLevelDirectory, G_FILE_TEST_IS_DIR))
            return topLevelDirectory;
    }
    // The tooling to run tests should provide the above environment variable with
    // the right value, but if that was not the case, then do an attempt to guess
    // it assuming that we were built in the standard WebKitBuild subdirectory.
    GUniquePtr<char*> parentPath(g_strsplit(currentExecutablePath().data(), "/WebKitBuild", -1));
    GUniquePtr<char> absoluteTopLevelPath(realpath(parentPath.get()[0], nullptr));
    return absoluteTopLevelPath.get();
}
#endif

} // namespace FileSystemImpl
} // namespace WTF
