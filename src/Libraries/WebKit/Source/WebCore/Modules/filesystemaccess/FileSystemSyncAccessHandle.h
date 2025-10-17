/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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

#include "ActiveDOMObject.h"
#include "BufferSource.h"
#include "ExceptionOr.h"
#include "FileHandle.h"
#include "FileSystemSyncAccessHandleIdentifier.h"
#include "IDLTypes.h"
#include <wtf/Deque.h>
#include <wtf/FileSystem.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class FileSystemFileHandle;
template<typename> class DOMPromiseDeferred;

class FileSystemSyncAccessHandle : public RefCountedAndCanMakeWeakPtr<FileSystemSyncAccessHandle>, public ActiveDOMObject {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    struct FilesystemReadWriteOptions {
        std::optional<unsigned long long> at;
    };

    static Ref<FileSystemSyncAccessHandle> create(ScriptExecutionContext&, FileSystemFileHandle&, FileSystemSyncAccessHandleIdentifier, FileHandle&&, uint64_t capacity);
    ~FileSystemSyncAccessHandle();

    ExceptionOr<void> truncate(unsigned long long size);
    ExceptionOr<unsigned long long> getSize();
    ExceptionOr<void> flush();
    ExceptionOr<void> close();
    ExceptionOr<unsigned long long> read(BufferSource&&, FilesystemReadWriteOptions);
    ExceptionOr<unsigned long long> write(BufferSource&&, FilesystemReadWriteOptions);
    void invalidate();

private:
    FileSystemSyncAccessHandle(ScriptExecutionContext&, FileSystemFileHandle&, FileSystemSyncAccessHandleIdentifier, FileHandle&&, uint64_t capacity);
    using CloseCallback = CompletionHandler<void(ExceptionOr<void>&&)>;
    enum class ShouldNotifyBackend : bool { No, Yes };
    void closeInternal(ShouldNotifyBackend);
    bool requestSpaceForNewSize(uint64_t newSize);
    bool requestSpaceForWrite(uint64_t writeOffset, uint64_t writeLength);

    // ActiveDOMObject.
    void stop() final;

    Ref<FileSystemFileHandle> m_source;
    FileSystemSyncAccessHandleIdentifier m_identifier;
    FileHandle m_file;
    bool m_isClosed { false };
    uint64_t m_capacity;
};

} // namespace WebCore
