/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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

#include "FileSystemStorageConnection.h"
#include "StorageConnection.h"
#include "StorageEstimate.h"
#include "StorageProvider.h"

namespace WebCore {

class DummyStorageProvider final : public StorageProvider {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(DummyStorageProvider, WEBCORE_EXPORT);
public:
    DummyStorageProvider() = default;

private:
    class DummyStorageConnection final : public StorageConnection {
    public:
        static Ref<DummyStorageConnection> create()
        {
            return adoptRef(*new DummyStorageConnection());
        }

        void getPersisted(ClientOrigin&&, StorageConnection::PersistCallback&& completionHandler) final
        {
            completionHandler(false);
        }

        void persist(const ClientOrigin&, StorageConnection::PersistCallback&& completionHandler) final
        {
            completionHandler(false);
        }

        void fileSystemGetDirectory(ClientOrigin&&, StorageConnection::GetDirectoryCallback&& completionHandler) final
        {
            completionHandler(Exception { ExceptionCode::NotSupportedError });
        }

        void getEstimate(ClientOrigin&&, GetEstimateCallback&& completionHandler) final
        {
            completionHandler(Exception { ExceptionCode::NotSupportedError });
        }
    };

    StorageConnection& storageConnection() final
    {
        if (!m_connection)
            m_connection = DummyStorageConnection::create();

        return *m_connection;
    }

    String ensureMediaKeysStorageDirectoryForOrigin(const SecurityOriginData& origin) final
    {
        if (m_mediaKeysStorageDirectory.isEmpty())
            return emptyString();

        auto originDirectory = FileSystem::pathByAppendingComponent(m_mediaKeysStorageDirectory, origin.databaseIdentifier());
        FileSystem::makeAllDirectories(originDirectory);
        return originDirectory;
    }

    void setMediaKeysStorageDirectory(const String& directory) final
    {
        m_mediaKeysStorageDirectory = directory;
    }

    RefPtr<DummyStorageConnection> m_connection;
    String m_mediaKeysStorageDirectory;
};

} // namespace WebCore
