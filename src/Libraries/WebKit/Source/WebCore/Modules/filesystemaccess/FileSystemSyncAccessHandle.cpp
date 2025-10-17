/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
#include "FileSystemSyncAccessHandle.h"

#include "BufferSource.h"
#include "FileSystemFileHandle.h"
#include "JSDOMPromiseDeferred.h"
#include <wtf/CompletionHandler.h>

namespace WebCore {

Ref<FileSystemSyncAccessHandle> FileSystemSyncAccessHandle::create(ScriptExecutionContext& context, FileSystemFileHandle& source, FileSystemSyncAccessHandleIdentifier identifier, FileHandle&& file, uint64_t capacity)
{
    auto handle = adoptRef(*new FileSystemSyncAccessHandle(context, source, identifier, WTFMove(file), capacity));
    handle->suspendIfNeeded();
    return handle;
}

FileSystemSyncAccessHandle::FileSystemSyncAccessHandle(ScriptExecutionContext& context, FileSystemFileHandle& source, FileSystemSyncAccessHandleIdentifier identifier, FileHandle&& file, uint64_t capacity)
    : ActiveDOMObject(&context)
    , m_source(source)
    , m_identifier(identifier)
    , m_file(WTFMove(file))
    , m_capacity(capacity)
{
    ASSERT(m_file);

    m_source->registerSyncAccessHandle(m_identifier, *this);
}

FileSystemSyncAccessHandle::~FileSystemSyncAccessHandle()
{
    m_source->unregisterSyncAccessHandle(m_identifier);
    closeInternal(ShouldNotifyBackend::Yes);
}

// https://fs.spec.whatwg.org/#dom-filesystemsyncaccesshandle-truncate
ExceptionOr<void> FileSystemSyncAccessHandle::truncate(unsigned long long size)
{
    if (m_isClosed)
        return Exception { ExceptionCode::InvalidStateError, "AccessHandle is closed"_s };

    auto oldSize = FileSystem::fileSize(m_file.handle());
    if (!oldSize)
        return Exception { ExceptionCode::InvalidStateError, "Failed to get current size"_s };

    if (size > *oldSize && !requestSpaceForNewSize(size))
        return Exception { ExceptionCode::QuotaExceededError };

    auto oldOffset = FileSystem::seekFile(m_file.handle(), 0, FileSystem::FileSeekOrigin::Current);
    if (oldOffset < 0)
        return Exception { ExceptionCode::InvalidStateError, "Failed to get current offset"_s };

    if (FileSystem::truncateFile(m_file.handle(), size)) {
        if (static_cast<uint64_t>(oldOffset) > size)
            FileSystem::seekFile(m_file.handle(), size, FileSystem::FileSeekOrigin::Beginning);

        return { };
    }

    return Exception { ExceptionCode::InvalidStateError, "Failed to truncate file"_s };
}

ExceptionOr<unsigned long long> FileSystemSyncAccessHandle::getSize()
{
    if (m_isClosed)
        return Exception { ExceptionCode::InvalidStateError, "AccessHandle is closed"_s };

    auto result = FileSystem::fileSize(m_file.handle());
    return result ? ExceptionOr<unsigned long long> { result.value() } : Exception { ExceptionCode::InvalidStateError, "Failed to get file size"_s };
}

ExceptionOr<void> FileSystemSyncAccessHandle::flush()
{
    if (m_isClosed)
        return Exception { ExceptionCode::InvalidStateError, "AccessHandle is closed"_s };

    bool succeeded = FileSystem::flushFile(m_file.handle());
    return succeeded ? ExceptionOr<void> { } : Exception { ExceptionCode::InvalidStateError, "Failed to flush file"_s };
}

ExceptionOr<void> FileSystemSyncAccessHandle::close()
{
    if (m_isClosed)
        return { };

    closeInternal(ShouldNotifyBackend::Yes);
    return { };
}

void FileSystemSyncAccessHandle::closeInternal(ShouldNotifyBackend shouldNotifyBackend)
{
    if (m_isClosed)
        return;

    m_isClosed = true;
    ASSERT(m_file);
    m_file = { };

    if (shouldNotifyBackend == ShouldNotifyBackend::Yes)
        m_source->closeSyncAccessHandle(m_identifier);
}

ExceptionOr<unsigned long long> FileSystemSyncAccessHandle::read(BufferSource&& buffer, FileSystemSyncAccessHandle::FilesystemReadWriteOptions options)
{
    if (m_isClosed)
        return Exception { ExceptionCode::InvalidStateError, "AccessHandle is closed"_s };

    if (options.at) {
        auto result = FileSystem::seekFile(m_file.handle(), options.at.value(), FileSystem::FileSeekOrigin::Beginning);
        if (result == -1)
            return Exception { ExceptionCode::InvalidStateError, "Failed to read at offset"_s };
    }

    int result = FileSystem::readFromFile(m_file.handle(), buffer.mutableSpan());
    if (result == -1)
        return Exception { ExceptionCode::InvalidStateError, "Failed to read from file"_s };

    return result;
}

ExceptionOr<unsigned long long> FileSystemSyncAccessHandle::write(BufferSource&& buffer, FileSystemSyncAccessHandle::FilesystemReadWriteOptions options)
{
    if (m_isClosed)
        return Exception { ExceptionCode::InvalidStateError, "AccessHandle is closed"_s };

    if (options.at) {
        auto result = FileSystem::seekFile(m_file.handle(), options.at.value(), FileSystem::FileSeekOrigin::Beginning);
        if (result == -1)
            return Exception { ExceptionCode::InvalidStateError, "Failed to write at offset"_s };
    } else {
        auto result = FileSystem::seekFile(m_file.handle(), 0, FileSystem::FileSeekOrigin::Current);
        if (result == -1)
            return Exception { ExceptionCode::InvalidStateError, "Failed to get offset"_s };
        options.at = result;
    }

    if (!requestSpaceForWrite(*options.at, buffer.length()))
        return Exception { ExceptionCode::QuotaExceededError };

    int result = FileSystem::writeToFile(m_file.handle(), buffer.span());
    if (result == -1)
        return Exception { ExceptionCode::InvalidStateError, "Failed to write to file"_s };

    return result;
}

void FileSystemSyncAccessHandle::stop()
{
    closeInternal(ShouldNotifyBackend::Yes);
}

void FileSystemSyncAccessHandle::invalidate()
{
    // Invalidation is initiated by backend.
    closeInternal(ShouldNotifyBackend::No);
}

bool FileSystemSyncAccessHandle::requestSpaceForNewSize(uint64_t newSize)
{
    if (newSize <= m_capacity)
        return true;

    auto newCapacity = m_source->requestNewCapacityForSyncAccessHandle(m_identifier, (uint64_t)newSize);
    if (newCapacity)
        m_capacity = *newCapacity;

    return newSize <= m_capacity;
}

bool FileSystemSyncAccessHandle::requestSpaceForWrite(uint64_t writeOffset, uint64_t writeLength)
{
    CheckedUint64 newSize = writeOffset;
    newSize += writeLength;
    if (newSize.hasOverflowed())
        return false;

    return requestSpaceForNewSize(newSize);
}

} // namespace WebCore
