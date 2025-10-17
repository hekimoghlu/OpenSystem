/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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

#include "FileSystemStorageHandle.h"
#include <WebCore/FileSystemHandleIdentifier.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class FileSystemStorageHandle;
class FileSystemStorageHandleRegistry;

class FileSystemStorageManager final : public RefCountedAndCanMakeWeakPtr<FileSystemStorageManager> {
    WTF_MAKE_TZONE_ALLOCATED(FileSystemStorageManager);
public:
    using QuotaCheckFunction = Function<void(uint64_t spaceRequested, CompletionHandler<void(bool)>&&)>;
    static Ref<FileSystemStorageManager> create(String&& path, FileSystemStorageHandleRegistry&, QuotaCheckFunction&&);
    ~FileSystemStorageManager();

    bool isActive() const;
    uint64_t allocatedUnusedCapacity() const;
    Expected<WebCore::FileSystemHandleIdentifier, FileSystemStorageError> createHandle(IPC::Connection::UniqueID, FileSystemStorageHandle::Type, String&& path, String&& name, bool createIfNecessary);
    const String& getPath(WebCore::FileSystemHandleIdentifier);
    FileSystemStorageHandle::Type getType(WebCore::FileSystemHandleIdentifier);
    void closeHandle(FileSystemStorageHandle&);
    void connectionClosed(IPC::Connection::UniqueID);
    Expected<WebCore::FileSystemHandleIdentifier, FileSystemStorageError> getDirectory(IPC::Connection::UniqueID);
    bool acquireLockForFile(const String& path, WebCore::FileSystemHandleIdentifier);
    bool releaseLockForFile(const String& path, WebCore::FileSystemHandleIdentifier);
    void requestSpace(uint64_t spaceRequested, CompletionHandler<void(bool)>&&);

private:
    FileSystemStorageManager(String&& path, FileSystemStorageHandleRegistry&, QuotaCheckFunction&&);

    void close();

    String m_path;
    WeakPtr<FileSystemStorageHandleRegistry> m_registry;
    QuotaCheckFunction m_quotaCheckFunction;
    HashMap<IPC::Connection::UniqueID, HashSet<WebCore::FileSystemHandleIdentifier>> m_handlesByConnection;
    HashMap<WebCore::FileSystemHandleIdentifier, RefPtr<FileSystemStorageHandle>> m_handles;
    HashMap<String, WebCore::FileSystemHandleIdentifier> m_lockMap;
};

} // namespace WebKit
