/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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

#include "FileHandle.h"
#include "FileSystemHandleIdentifier.h"
#include "FileSystemSyncAccessHandleIdentifier.h"
#include "FileSystemWriteCloseReason.h"
#include "FileSystemWriteCommandType.h"
#include "ProcessQualified.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/CompletionHandler.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class FileSystemDirectoryHandle;
class FileSystemFileHandle;
class FileHandle;
class FileSystemHandleCloseScope;
class FileSystemSyncAccessHandle;
template<typename> class ExceptionOr;

class FileSystemStorageConnection : public ThreadSafeRefCounted<FileSystemStorageConnection> {
public:
    virtual ~FileSystemStorageConnection() { }

    using SameEntryCallback = CompletionHandler<void(ExceptionOr<bool>&&)>;
    using GetHandleCallback = CompletionHandler<void(ExceptionOr<Ref<FileSystemHandleCloseScope>>&&)>;
    using ResolveCallback = CompletionHandler<void(ExceptionOr<Vector<String>>&&)>;
    struct SyncAccessHandleInfo {
        FileSystemSyncAccessHandleIdentifier identifier;
        FileHandle file;
        uint64_t capacity { 0 };
        SyncAccessHandleInfo isolatedCopy() && { return { identifier, WTFMove(file).isolatedCopy(), capacity }; }
    };
    using GetAccessHandleCallback = CompletionHandler<void(ExceptionOr<SyncAccessHandleInfo>&&)>;
    using VoidCallback = CompletionHandler<void(ExceptionOr<void>&&)>;
    using EmptyCallback = CompletionHandler<void()>;
    using GetHandleNamesCallback = CompletionHandler<void(ExceptionOr<Vector<String>>&&)>;
    using StringCallback = CompletionHandler<void(ExceptionOr<String>&&)>;
    using RequestCapacityCallback = CompletionHandler<void(std::optional<uint64_t>&&)>;

    virtual bool isWorker() const { return false; }
    virtual void closeHandle(FileSystemHandleIdentifier) = 0;
    virtual void isSameEntry(FileSystemHandleIdentifier, FileSystemHandleIdentifier, SameEntryCallback&&) = 0;
    virtual void move(FileSystemHandleIdentifier, FileSystemHandleIdentifier, const String& newName, VoidCallback&&) = 0;
    virtual void getFileHandle(FileSystemHandleIdentifier, const String& name, bool createIfNecessary, GetHandleCallback&&) = 0;
    virtual void getDirectoryHandle(FileSystemHandleIdentifier, const String& name, bool createIfNecessary, GetHandleCallback&&) = 0;
    virtual void removeEntry(FileSystemHandleIdentifier, const String& name, bool deleteRecursively, VoidCallback&&) = 0;
    virtual void resolve(FileSystemHandleIdentifier, FileSystemHandleIdentifier, ResolveCallback&&) = 0;
    virtual void getFile(FileSystemHandleIdentifier, StringCallback&&) = 0;
    virtual void createSyncAccessHandle(FileSystemHandleIdentifier, GetAccessHandleCallback&&) = 0;
    virtual void closeSyncAccessHandle(FileSystemHandleIdentifier, FileSystemSyncAccessHandleIdentifier, EmptyCallback&&) = 0;
    virtual void requestNewCapacityForSyncAccessHandle(FileSystemHandleIdentifier, FileSystemSyncAccessHandleIdentifier, uint64_t newCapacity, RequestCapacityCallback&&) = 0;
    virtual void registerSyncAccessHandle(FileSystemSyncAccessHandleIdentifier, ScriptExecutionContextIdentifier) = 0;
    virtual void unregisterSyncAccessHandle(FileSystemSyncAccessHandleIdentifier) = 0;
    virtual void invalidateAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier) = 0;
    virtual void createWritable(FileSystemHandleIdentifier, bool keepExistingData, VoidCallback&&) = 0;
    virtual void closeWritable(FileSystemHandleIdentifier, FileSystemWriteCloseReason, VoidCallback&&) = 0;
    virtual void executeCommandForWritable(FileSystemHandleIdentifier, FileSystemWriteCommandType, std::optional<uint64_t> position, std::optional<uint64_t> size, std::span<const uint8_t> dataBytes, bool hasDataError, VoidCallback&&) = 0;
    virtual void getHandleNames(FileSystemHandleIdentifier, GetHandleNamesCallback&&) = 0;
    virtual void getHandle(FileSystemHandleIdentifier, const String& name, GetHandleCallback&&) = 0;
};

} // namespace WebCore
