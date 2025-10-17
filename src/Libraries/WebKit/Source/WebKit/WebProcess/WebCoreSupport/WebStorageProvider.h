/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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
#pragma once

#include "WebStorageConnection.h"
#include <WebCore/StorageProvider.h>
#include <WebCore/StorageUtilities.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

class WebStorageProvider final : public WebCore::StorageProvider {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WebStorageProvider);
public:
    WebStorageProvider(const String& mediaKeysStorageDirectory, FileSystem::Salt mediaKeysStorageSalt)
        : m_mediaKeysStorageDirectory(mediaKeysStorageDirectory)
        , m_mediaKeysStorageSalt(mediaKeysStorageSalt)
    {
    }

private:
    WebCore::StorageConnection& storageConnection() final
    {
        ASSERT(WTF::isMainRunLoop());

        if (!m_connection)
            m_connection = WebStorageConnection::create();
        
        return *m_connection;
    }

    String ensureMediaKeysStorageDirectoryForOrigin(const WebCore::SecurityOriginData& origin) final
    {
        if (m_mediaKeysStorageDirectory.isEmpty())
            return emptyString();

        auto originDirectoryName = WebCore::StorageUtilities::encodeSecurityOriginForFileName(m_mediaKeysStorageSalt, origin);
        auto originDirectory = FileSystem::pathByAppendingComponent(m_mediaKeysStorageDirectory, originDirectoryName);
        WebCore::StorageUtilities::writeOriginToFile(FileSystem::pathByAppendingComponent(originDirectory, "origin"_s), WebCore::ClientOrigin { origin, origin });
        return originDirectory;
    }

    void setMediaKeysStorageDirectory(const String&) final
    {
        RELEASE_ASSERT_NOT_REACHED();
    }

    RefPtr<WebStorageConnection> m_connection;
    String m_mediaKeysStorageDirectory;
    FileSystem::Salt m_mediaKeysStorageSalt;
};

} // namespace WebKit
