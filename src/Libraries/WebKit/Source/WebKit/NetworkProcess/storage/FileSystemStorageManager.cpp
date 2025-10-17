/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#include "FileSystemStorageManager.h"

#include "FileSystemStorageError.h"
#include "FileSystemStorageHandleRegistry.h"
#include "WebFileSystemStorageConnectionMessages.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FileSystemStorageManager);

Ref<FileSystemStorageManager> FileSystemStorageManager::create(String&& path, FileSystemStorageHandleRegistry& registry, QuotaCheckFunction&& quotaCheckFunction)
{
    return adoptRef(*new FileSystemStorageManager(WTFMove(path), registry, WTFMove(quotaCheckFunction)));
}

FileSystemStorageManager::FileSystemStorageManager(String&& path, FileSystemStorageHandleRegistry& registry, QuotaCheckFunction&& quotaCheckFunction)
    : m_path(WTFMove(path))
    , m_registry(registry)
    , m_quotaCheckFunction(WTFMove(quotaCheckFunction))
{
    ASSERT(!RunLoop::isMain());
}

FileSystemStorageManager::~FileSystemStorageManager()
{
    ASSERT(!RunLoop::isMain());

    close();
}

bool FileSystemStorageManager::isActive() const
{
    return !m_handles.isEmpty();
}

uint64_t FileSystemStorageManager::allocatedUnusedCapacity() const
{
    CheckedUint64 result = 0;
    for (auto& handle : m_handles.values())
        result += handle->allocatedUnusedCapacity();

    if (result.hasOverflowed())
        return 0;

    return result;
}

Expected<WebCore::FileSystemHandleIdentifier, FileSystemStorageError> FileSystemStorageManager::createHandle(IPC::Connection::UniqueID connection, FileSystemStorageHandle::Type type, String&& path, String&& name, bool createIfNecessary)
{
    ASSERT(!RunLoop::isMain());

    if (path.isEmpty())
        return makeUnexpected(FileSystemStorageError::Unknown);

    auto fileExists = FileSystem::fileExists(path);
    if (!createIfNecessary && !fileExists)
        return makeUnexpected(FileSystemStorageError::FileNotFound);

    if (fileExists) {
        auto existingFileType = FileSystem::fileType(path);
        if (!existingFileType)
            return makeUnexpected(FileSystemStorageError::Unknown);

        auto existingHandleType = (existingFileType.value() == FileSystem::FileType::Regular) ? FileSystemStorageHandle::Type::File : FileSystemStorageHandle::Type::Directory;
        if (type == FileSystemStorageHandle::Type::Any)
            type = existingHandleType;
        else {
            // Requesting type and existing type should be a match.
            if (existingHandleType != type)
                return makeUnexpected(FileSystemStorageError::TypeMismatch);
        }
    }

    RefPtr newHandle = FileSystemStorageHandle::create(*this, type, WTFMove(path), WTFMove(name));
    if (!newHandle)
        return makeUnexpected(FileSystemStorageError::Unknown);
    auto newHandleIdentifier = newHandle->identifier();
    m_handlesByConnection.ensure(connection, [&] {
        return HashSet<WebCore::FileSystemHandleIdentifier> { };
    }).iterator->value.add(newHandleIdentifier);
    if (RefPtr registry = m_registry.get())
        registry->registerHandle(newHandleIdentifier, *newHandle);
    m_handles.add(newHandleIdentifier, WTFMove(newHandle));
    return newHandleIdentifier;
}

const String& FileSystemStorageManager::getPath(WebCore::FileSystemHandleIdentifier identifier)
{
    auto handle = m_handles.find(identifier);
    return handle == m_handles.end() ? emptyString() : handle->value->path();
}

FileSystemStorageHandle::Type FileSystemStorageManager::getType(WebCore::FileSystemHandleIdentifier identifier)
{
    auto handle = m_handles.find(identifier);
    return handle == m_handles.end() ? FileSystemStorageHandle::Type::Any : handle->value->type();
}

void FileSystemStorageManager::closeHandle(FileSystemStorageHandle& handle)
{
    auto identifier = handle.identifier();
    auto takenHandle = m_handles.take(identifier);
    ASSERT(takenHandle.get() == &handle);
    for (auto& handles : m_handlesByConnection.values()) {
        if (handles.remove(identifier))
            break;
    }
    if (RefPtr registry = m_registry.get())
        registry->unregisterHandle(identifier);
}

void FileSystemStorageManager::connectionClosed(IPC::Connection::UniqueID connection)
{
    ASSERT(!RunLoop::isMain());

    auto connectionHandles = m_handlesByConnection.find(connection);
    if (connectionHandles == m_handlesByConnection.end())
        return;

    auto identifiers = connectionHandles->value;
    for (auto identifier : identifiers) {
        m_handles.remove(identifier);
        if (RefPtr registry = m_registry.get())
            registry->unregisterHandle(identifier);
    }

    m_lockMap.removeIf([&identifiers](auto& entry) {
        return identifiers.contains(entry.value);
    });

    m_handlesByConnection.remove(connectionHandles);
}

Expected<WebCore::FileSystemHandleIdentifier, FileSystemStorageError> FileSystemStorageManager::getDirectory(IPC::Connection::UniqueID connection)
{
    ASSERT(!RunLoop::isMain());

    return createHandle(connection, FileSystemStorageHandle::Type::Directory, String { m_path }, { }, true);
}

bool FileSystemStorageManager::acquireLockForFile(const String& path, WebCore::FileSystemHandleIdentifier identifier)
{
    if (m_lockMap.contains(path))
        return false;

    m_lockMap.add(path, identifier);
    return true;
}

bool FileSystemStorageManager::releaseLockForFile(const String& path, WebCore::FileSystemHandleIdentifier identifier)
{
    if (auto lockedByIdentifier = m_lockMap.get(path); lockedByIdentifier == identifier) {
        m_lockMap.remove(path);
        return true;
    }

    return false;
}

void FileSystemStorageManager::close()
{
    ASSERT(!RunLoop::isMain());

    for (auto& [connectionID, identifiers] : m_handlesByConnection) {
        for (auto identifier : identifiers) {
            auto takenHandle = m_handles.take(identifier);
            if (RefPtr registry = m_registry.get())
                registry->unregisterHandle(identifier);

            // Send message to web process to invalidate active sync access handle.
            if (auto accessHandleIdentifier = takenHandle->activeSyncAccessHandle())
                IPC::Connection::send(connectionID, Messages::WebFileSystemStorageConnection::InvalidateAccessHandle(*accessHandleIdentifier), 0);
        }
    }

    ASSERT(m_handles.isEmpty());
    m_handlesByConnection.clear();
    m_lockMap.clear();
}

void FileSystemStorageManager::requestSpace(uint64_t size, CompletionHandler<void(bool)>&& completionHandler)
{
    m_quotaCheckFunction(size, WTFMove(completionHandler));
}

} // namespace WebKit
