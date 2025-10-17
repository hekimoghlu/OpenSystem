/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 3, 2024.
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
#include "WebMediaKeyStorageManager.h"

#include "WebProcessDataStoreParameters.h"
#include <WebCore/SecurityOrigin.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/FileSystem.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebMediaKeyStorageManager);

void WebMediaKeyStorageManager::setWebsiteDataStore(const WebProcessDataStoreParameters& parameters)
{
    m_mediaKeyStorageDirectory = parameters.mediaKeyStorageDirectory;
}

ASCIILiteral WebMediaKeyStorageManager::supplementName()
{
    return "WebMediaKeyStorageManager"_s;
}

String WebMediaKeyStorageManager::mediaKeyStorageDirectoryForOrigin(const SecurityOriginData& originData)
{
    if (!m_mediaKeyStorageDirectory)
        return emptyString();

    return FileSystem::pathByAppendingComponent(m_mediaKeyStorageDirectory, originData.databaseIdentifier());
}

Vector<SecurityOriginData> WebMediaKeyStorageManager::getMediaKeyOrigins()
{
    Vector<SecurityOriginData> results;

    if (m_mediaKeyStorageDirectory.isEmpty())
        return results;

    for (auto& identifier : FileSystem::listDirectory(m_mediaKeyStorageDirectory)) {
        if (auto securityOrigin = SecurityOriginData::fromDatabaseIdentifier(identifier))
            results.append(*securityOrigin);
    }

    return results;
}

static void removeAllMediaKeyStorageForOriginPath(const String& originPath, WallTime startDate, WallTime endDate)
{
    Vector<String> mediaKeyNames = FileSystem::listDirectory(originPath);

    for (const auto& mediaKeyName : mediaKeyNames) {
        auto mediaKeyPath = FileSystem::pathByAppendingComponent(originPath, mediaKeyName);
        String mediaKeyFile = FileSystem::pathByAppendingComponent(mediaKeyPath, "SecureStop.plist"_s);

        if (!FileSystem::fileExists(mediaKeyFile))
            continue;

        auto modificationTime = FileSystem::fileModificationTime(mediaKeyFile);
        if (!modificationTime)
            continue;
        if (modificationTime.value() < startDate || modificationTime.value() > endDate)
            continue;

        FileSystem::deleteFile(mediaKeyFile);
        FileSystem::deleteEmptyDirectory(mediaKeyPath);
    }
    
    FileSystem::deleteEmptyDirectory(originPath);
}

void WebMediaKeyStorageManager::deleteMediaKeyEntriesForOrigin(const SecurityOriginData& originData)
{
    if (m_mediaKeyStorageDirectory.isEmpty())
        return;

    String originPath = mediaKeyStorageDirectoryForOrigin(originData);
    removeAllMediaKeyStorageForOriginPath(originPath, -WallTime::infinity(), WallTime::infinity());
}

void WebMediaKeyStorageManager::deleteMediaKeyEntriesModifiedBetweenDates(WallTime startDate, WallTime endDate)
{
    if (m_mediaKeyStorageDirectory.isEmpty())
        return;

    Vector<String> originNames = FileSystem::listDirectory(m_mediaKeyStorageDirectory);
    for (auto& originName : originNames)
        removeAllMediaKeyStorageForOriginPath(FileSystem::pathByAppendingComponent(m_mediaKeyStorageDirectory, originName), startDate, endDate);
}

void WebMediaKeyStorageManager::deleteAllMediaKeyEntries()
{
    if (m_mediaKeyStorageDirectory.isEmpty())
        return;

    Vector<String> originNames = FileSystem::listDirectory(m_mediaKeyStorageDirectory);
    for (auto& originName : originNames)
        removeAllMediaKeyStorageForOriginPath(FileSystem::pathByAppendingComponent(m_mediaKeyStorageDirectory, originName), -WallTime::infinity(), WallTime::infinity());
}

}
