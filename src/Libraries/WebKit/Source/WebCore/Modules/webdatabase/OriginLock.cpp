/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#include "OriginLock.h"

namespace WebCore {

static String lockFileNameForPath(const String& originPath)
{
    return FileSystem::pathByAppendingComponent(originPath, ".lock"_s);
}

OriginLock::OriginLock(const String& originPath)
    : m_lockFileName(lockFileNameForPath(originPath).isolatedCopy())
{
}

OriginLock::~OriginLock() = default;

void OriginLock::lock() WTF_IGNORES_THREAD_SAFETY_ANALYSIS
{
    m_mutex.lock();

#if USE(FILE_LOCK)
    m_lockHandle = FileSystem::openAndLockFile(m_lockFileName, FileSystem::FileOpenMode::Truncate);
    if (m_lockHandle == FileSystem::invalidPlatformFileHandle) {
        // The only way we can get here is if the directory containing the lock
        // has been deleted or we were given a path to a non-existant directory.
        // In that case, there's nothing we can do but cleanup and return.
        m_mutex.unlock();
        return;
    }
#endif
}

void OriginLock::unlock() WTF_IGNORES_THREAD_SAFETY_ANALYSIS
{
#if USE(FILE_LOCK)
    // If the file descriptor was uninitialized, then that means the directory
    // containing the lock has been deleted before we opened the lock file, or
    // we were given a path to a non-existant directory. Which, in turn, means
    // that there's nothing to unlock.
    if (m_lockHandle == FileSystem::invalidPlatformFileHandle)
        return;

    FileSystem::unlockAndCloseFile(m_lockHandle);
    m_lockHandle = FileSystem::invalidPlatformFileHandle;
#endif

    m_mutex.unlock();
}

void OriginLock::deleteLockFile(const String& originPath)
{
#if USE(FILE_LOCK)
    FileSystem::deleteFile(lockFileNameForPath(originPath));
#else
    UNUSED_PARAM(originPath);
#endif
}

} // namespace WebCore
