/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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

#include "Connection.h"
#include "FileSystemSyncAccessHandleInfo.h"
#include <WebCore/FileSystemHandleIdentifier.h>
#include <WebCore/FileSystemSyncAccessHandleIdentifier.h>
#include <wtf/Identified.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class SharedFileHandle;
}

namespace WebCore {
enum class FileSystemWriteCloseReason : bool;
enum class FileSystemWriteCommandType : uint8_t;
}

namespace WebKit {

class FileSystemStorageManager;
enum class FileSystemStorageError : uint8_t;

class FileSystemStorageHandle final : public RefCounted<FileSystemStorageHandle>, public CanMakeWeakPtr<FileSystemStorageHandle, WeakPtrFactoryInitialization::Eager>, public Identified<WebCore::FileSystemHandleIdentifier> {
    WTF_MAKE_TZONE_ALLOCATED(FileSystemStorageHandle);
public:
    enum class Type : uint8_t { File, Directory, Any };
    static RefPtr<FileSystemStorageHandle> create(FileSystemStorageManager&, Type, String&& path, String&& name);

    const String& path() const { return m_path; }
    Type type() const { return m_type; }
    uint64_t allocatedUnusedCapacity();

    void close();
    bool isSameEntry(WebCore::FileSystemHandleIdentifier);
    std::optional<FileSystemStorageError> move(WebCore::FileSystemHandleIdentifier, const String& newName);
    Expected<WebCore::FileSystemHandleIdentifier, FileSystemStorageError> getFileHandle(IPC::Connection::UniqueID, String&& name, bool createIfNecessary);
    Expected<WebCore::FileSystemHandleIdentifier, FileSystemStorageError> getDirectoryHandle(IPC::Connection::UniqueID, String&& name, bool createIfNecessary);
    std::optional<FileSystemStorageError> removeEntry(const String& name, bool deleteRecursively);
    Expected<Vector<String>, FileSystemStorageError> resolve(WebCore::FileSystemHandleIdentifier);
    Expected<Vector<String>, FileSystemStorageError> getHandleNames();
    Expected<std::pair<WebCore::FileSystemHandleIdentifier, bool>, FileSystemStorageError> getHandle(IPC::Connection::UniqueID, String&& name);
    void requestNewCapacityForSyncAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier, uint64_t newCapacity, CompletionHandler<void(std::optional<uint64_t>)>&&);

    Expected<FileSystemSyncAccessHandleInfo, FileSystemStorageError> createSyncAccessHandle();
    std::optional<FileSystemStorageError> closeSyncAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier);
    std::optional<WebCore::FileSystemSyncAccessHandleIdentifier> activeSyncAccessHandle();

    std::optional<FileSystemStorageError> createWritable(bool keepExistingData);
    std::optional<FileSystemStorageError> closeWritable(WebCore::FileSystemWriteCloseReason);
    std::optional<FileSystemStorageError> executeCommandForWritable(WebCore::FileSystemWriteCommandType, std::optional<uint64_t> position, std::optional<uint64_t> size, std::span<const uint8_t> dataBytes, bool hasDataError);

private:
    FileSystemStorageHandle(FileSystemStorageManager&, Type, String&& path, String&& name);
    Expected<WebCore::FileSystemHandleIdentifier, FileSystemStorageError> requestCreateHandle(IPC::Connection::UniqueID, Type, String&& name, bool createIfNecessary);
    bool isActiveSyncAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier);
    std::optional<FileSystemStorageError> executeCommandForWritableInternal(WebCore::FileSystemWriteCommandType, std::optional<uint64_t> position, std::optional<uint64_t> size, std::span<const uint8_t> dataBytes, bool hasDataError);

    WeakPtr<FileSystemStorageManager> m_manager;
    Type m_type;
    String m_path;
    String m_name;
    struct SyncAccessHandleInfo {
        WebCore::FileSystemSyncAccessHandleIdentifier identifier;
        uint64_t capacity { 0 };
    };
    std::optional<SyncAccessHandleInfo> m_activeSyncAccessHandle;
    // FIXME: Support multiple writable streams.
    WebCore::FileHandle m_activeWritableFile;
};

} // namespace WebKit
