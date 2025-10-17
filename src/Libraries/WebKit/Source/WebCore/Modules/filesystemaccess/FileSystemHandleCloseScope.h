/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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
#include <wtf/RunLoop.h>

namespace WebCore {

class FileSystemHandleCloseScope : public ThreadSafeRefCounted<FileSystemHandleCloseScope, WTF::DestructionThread::MainRunLoop> {
public:
    static Ref<FileSystemHandleCloseScope> create(FileSystemHandleIdentifier identifier, bool isDirectory, FileSystemStorageConnection& connection)
    {
        return adoptRef(*new FileSystemHandleCloseScope(identifier, isDirectory, connection));
    }

    ~FileSystemHandleCloseScope()
    {
        ASSERT(RunLoop::isMain());

        if (m_identifier)
            m_connection->closeHandle(*m_identifier);
    }

    std::pair<FileSystemHandleIdentifier, bool> release()
    {
        Locker locker { m_lock };
        ASSERT_WITH_MESSAGE(!!m_identifier, "FileSystemHandleCloseScope should not be released more than once");
        return { *std::exchange(m_identifier, std::nullopt), m_isDirectory };
    }

private:
    FileSystemHandleCloseScope(FileSystemHandleIdentifier identifier, bool isDirectory, FileSystemStorageConnection& connection)
        : m_identifier(identifier)
        , m_isDirectory(isDirectory)
        , m_connection(Ref { connection })
    {
        ASSERT(RunLoop::isMain());
    }

    Lock m_lock;
    Markable<FileSystemHandleIdentifier> m_identifier WTF_GUARDED_BY_LOCK(m_lock);
    bool m_isDirectory;
    Ref<FileSystemStorageConnection> m_connection;
};

} // namespace WebCore
