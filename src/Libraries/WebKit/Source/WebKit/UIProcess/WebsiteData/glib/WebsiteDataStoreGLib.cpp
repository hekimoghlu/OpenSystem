/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
#include "WebsiteDataStore.h"

#include <wtf/FileSystem.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {

static String programName()
{
    if (auto* prgname = g_get_prgname())
        return String::fromUTF8(prgname);

#if PLATFORM(GTK)
    return "webkitgtk"_s;
#elif PLATFORM(WPE)
    return "wpe"_s;
#else
    return "WebKit"_s;
#endif
}

const String& WebsiteDataStore::defaultBaseCacheDirectory()
{
    static NeverDestroyed<String> baseCacheDirectory;
    static std::once_flag once;
    std::call_once(once, [] {

        baseCacheDirectory.get() = FileSystem::pathByAppendingComponent(FileSystem::userCacheDirectory(), programName());
    });
    return baseCacheDirectory;
}

const String& WebsiteDataStore::defaultBaseDataDirectory()
{
    static NeverDestroyed<String> baseDataDirectory;
    static std::once_flag once;
    std::call_once(once, [] {
        baseDataDirectory.get() = FileSystem::pathByAppendingComponent(FileSystem::userDataDirectory(), programName());
    });
    return baseDataDirectory;
}

String WebsiteDataStore::cacheDirectoryFileSystemRepresentation(const String& directoryName, const String& baseCacheDirectory, ShouldCreateDirectory)
{
    return FileSystem::pathByAppendingComponent(baseCacheDirectory.isNull() ? defaultBaseCacheDirectory() : baseCacheDirectory, directoryName);
}

String WebsiteDataStore::websiteDataDirectoryFileSystemRepresentation(const String& directoryName, const String& baseDataDirectory, ShouldCreateDirectory)
{
    return FileSystem::pathByAppendingComponent(baseDataDirectory.isNull() ? defaultBaseDataDirectory() : baseDataDirectory, directoryName);
}

UnifiedOriginStorageLevel WebsiteDataStore::defaultUnifiedOriginStorageLevel()
{
#if ENABLE(2022_GLIB_API)
    return UnifiedOriginStorageLevel::Basic;
#else
    return UnifiedOriginStorageLevel::None;
#endif
}

void WebsiteDataStore::platformInitialize()
{
}

void WebsiteDataStore::platformDestroy()
{
}

} // namespace WebKit
