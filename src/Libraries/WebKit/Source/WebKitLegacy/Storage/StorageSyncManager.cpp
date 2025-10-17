/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 28, 2023.
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
#include "StorageSyncManager.h"

#include "StorageThread.h"
#include <wtf/FileSystem.h>
#include <wtf/MainThread.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

Ref<StorageSyncManager> StorageSyncManager::create(const String& path)
{
    return adoptRef(*new StorageSyncManager(path));
}

StorageSyncManager::StorageSyncManager(const String& path)
    : m_thread(makeUnique<StorageThread>())
    , m_path(path.isolatedCopy())
{
    ASSERT(isMainThread());
    ASSERT(!m_path.isEmpty());
    m_thread->start();
}

StorageSyncManager::~StorageSyncManager()
{
    ASSERT(isMainThread());
    ASSERT(!m_thread);
}

// Called on a background thread.
String StorageSyncManager::fullDatabaseFilename(const String& databaseIdentifier)
{
    if (!FileSystem::makeAllDirectories(m_path)) {
        LOG_ERROR("Unabled to create LocalStorage database path %s", m_path.utf8().data());
        return String();
    }

    return FileSystem::pathByAppendingComponent(m_path, makeString(databaseIdentifier, ".localstorage"_s));
}

void StorageSyncManager::dispatch(Function<void ()>&& function)
{
    ASSERT(isMainThread());
    ASSERT(m_thread);

    if (m_thread)
        m_thread->dispatch(WTFMove(function));
}

void StorageSyncManager::close()
{
    ASSERT(isMainThread());

    if (m_thread) {
        m_thread->terminate();
        m_thread = nullptr;
    }
}

} // namespace WebCore
