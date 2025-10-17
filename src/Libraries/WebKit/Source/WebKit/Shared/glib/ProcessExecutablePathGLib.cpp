/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include "ProcessExecutablePath.h"

#include <glib.h>
#include <wtf/FileSystem.h>

namespace WebKit {

#if ENABLE(DEVELOPER_MODE)
static String getExecutablePath()
{
    CString executablePath = FileSystem::currentExecutablePath();
    if (!executablePath.isNull())
        return FileSystem::parentPath(FileSystem::stringFromFileSystemRepresentation(executablePath.data()));
    return { };
}
#endif

static String findWebKitProcess(const char* processName)
{
#if ENABLE(DEVELOPER_MODE)
    static const char* execDirectory = g_getenv("WEBKIT_EXEC_PATH");
    if (execDirectory) {
        String processPath = FileSystem::pathByAppendingComponent(FileSystem::stringFromFileSystemRepresentation(execDirectory), StringView::fromLatin1(processName));
        if (FileSystem::fileExists(processPath))
            return processPath;
    }

    static String executablePath = getExecutablePath();
    if (!executablePath.isNull()) {
        String processPath = FileSystem::pathByAppendingComponent(executablePath, StringView::fromLatin1(processName));
        if (FileSystem::fileExists(processPath))
            return processPath;
    }
#endif

    return FileSystem::pathByAppendingComponent(FileSystem::stringFromFileSystemRepresentation(PKGLIBEXECDIR), StringView::fromLatin1(processName));
}

String executablePathOfWebProcess()
{
#if PLATFORM(WPE)
    return findWebKitProcess("WPEWebProcess");
#else
    return findWebKitProcess("WebKitWebProcess");
#endif
}

String executablePathOfPluginProcess()
{
#if PLATFORM(WPE)
    return findWebKitProcess("WPEPluginProcess");
#else
    return findWebKitProcess("WebKitPluginProcess");
#endif
}

String executablePathOfNetworkProcess()
{
#if PLATFORM(WPE)
    return findWebKitProcess("WPENetworkProcess");
#else
    return findWebKitProcess("WebKitNetworkProcess");
#endif
}

#if ENABLE(GPU_PROCESS)
String executablePathOfGPUProcess()
{
#if PLATFORM(WPE)
    return findWebKitProcess("WPEGPUProcess");
#else
    return findWebKitProcess("WebKitGPUProcess");
#endif
}
#endif

} // namespace WebKit
