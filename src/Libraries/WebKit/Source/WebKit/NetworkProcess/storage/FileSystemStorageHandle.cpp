/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
#include "FileSystemStorageHandle.h"

#include "FileSystemStorageError.h"
#include "FileSystemStorageManager.h"
#include "SharedFileHandle.h"
#include <WebCore/FileSystemWriteCloseReason.h>
#include <WebCore/FileSystemWriteCommandType.h>
#include <wtf/Scope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

#if OS(WINDOWS)
constexpr char pathSeparator = '\\';
#else
constexpr char pathSeparator = '/';
#endif
constexpr uint64_t defaultInitialCapacity = 1 * MB;
constexpr uint64_t defaultMaxCapacityForExponentialGrowth = 256 * MB;
constexpr uint64_t defaultCapacityStep = 128 * MB;

WTF_MAKE_TZONE_ALLOCATED_IMPL(FileSystemStorageHandle);

RefPtr<FileSystemStorageHandle> FileSystemStorageHandle::create(FileSystemStorageManager& manager, Type type, String&& path, String&& name)
{
    bool canAccess = false;
    switch (type) {
    case FileSystemStorageHandle::Type::Directory:
        canAccess = FileSystem::makeAllDirectories(path);
        break;
    case FileSystemStorageHandle::Type::File:
        if (auto handle = FileSystem::openFile(path, FileSystem::FileOpenMode::ReadWrite); FileSystem::isHandleValid(handle)) {
            FileSystem::closeFile(handle);
            canAccess = true;
        }
        break;
    case FileSystemStorageHandle::Type::Any:
        ASSERT_NOT_REACHED();
    }

    if (!canAccess)
        return nullptr;

    return adoptRef(*new FileSystemStorageHandle(manager, type, WTFMove(path), WTFMove(name)));
}

FileSystemStorageHandle::FileSystemStorageHandle(FileSystemStorageManager& manager, Type type, String&& path, String&& name)
    : m_manager(manager)
    , m_type(type)
    , m_path(WTFMove(path))
    , m_name(WTFMove(name))
{
    ASSERT(!m_path.isEmpty());
}

void FileSystemStorageHandle::close()
{
    RefPtr manager = m_manager.get();
    if (!manager)
        return;

    if (m_activeSyncAccessHandle)
        closeSyncAccessHandle(m_activeSyncAccessHandle->identifier);

    closeWritable(WebCore::FileSystemWriteCloseReason::Aborted);
    manager->closeHandle(*this);
}

bool FileSystemStorageHandle::isSameEntry(WebCore::FileSystemHandleIdentifier identifier)
{
    RefPtr manager = m_manager.get();
    if (!manager)
        return false;

    auto path = manager->getPath(identifier);
    if (path.isEmpty())
        return false;

    return m_path == path;
}

static bool isValidFileName(const String& directory, const String& name)
{
    // https://fs.spec.whatwg.org/#valid-file-name
    if (name.isEmpty() || (name == "."_s) || (name == ".."_s) || name.contains(pathSeparator))
        return false;

    return FileSystem::pathFileName(FileSystem::pathByAppendingComponent(directory, name)) == name;
}

Expected<WebCore::FileSystemHandleIdentifier, FileSystemStorageError> FileSystemStorageHandle::requestCreateHandle(IPC::Connection::UniqueID connection, Type type, String&& name, bool createIfNecessary)
{
    if (m_type != FileSystemStorageHandle::Type::Directory)
        return makeUnexpected(FileSystemStorageError::TypeMismatch);

    RefPtr manager = m_manager.get();
    if (!manager)
        return makeUnexpected(FileSystemStorageError::Unknown);

    if (!isValidFileName(m_path, name))
        return makeUnexpected(FileSystemStorageError::InvalidName);

    auto path = FileSystem::pathByAppendingComponent(m_path, name);
    return manager->createHandle(connection, type, WTFMove(path), WTFMove(name), createIfNecessary);
}

Expected<WebCore::FileSystemHandleIdentifier, FileSystemStorageError> FileSystemStorageHandle::getFileHandle(IPC::Connection::UniqueID connection, String&& name, bool createIfNecessary)
{
    return requestCreateHandle(connection, FileSystemStorageHandle::Type::File, WTFMove(name), createIfNecessary);
}

Expected<WebCore::FileSystemHandleIdentifier, FileSystemStorageError> FileSystemStorageHandle::getDirectoryHandle(IPC::Connection::UniqueID connection, String&& name, bool createIfNecessary)
{
    return requestCreateHandle(connection, FileSystemStorageHandle::Type::Directory, WTFMove(name), createIfNecessary);
}

std::optional<FileSystemStorageError> FileSystemStorageHandle::removeEntry(const String& name, bool deleteRecursively)
{
    if (m_type != Type::Directory)
        return FileSystemStorageError::TypeMismatch;

    if (!isValidFileName(m_path, name))
        return FileSystemStorageError::InvalidName;

    auto path = FileSystem::pathByAppendingComponent(m_path, name);
    if (!FileSystem::fileExists(path))
        return FileSystemStorageError::FileNotFound;

    auto type = FileSystem::fileType(path);
    if (!type)
        return FileSystemStorageError::TypeMismatch;

    std::optional<FileSystemStorageError> result;
    switch (type.value()) {
    case FileSystem::FileType::Regular:
        if (!FileSystem::deleteFile(path))
            result = FileSystemStorageError::Unknown;
        break;
    case FileSystem::FileType::Directory:
        if (!deleteRecursively) {
            if (!FileSystem::deleteEmptyDirectory(path))
                result = FileSystemStorageError::Unknown;
        } else if (!FileSystem::deleteNonEmptyDirectory(path))
            result = FileSystemStorageError::Unknown;
        break;
    case FileSystem::FileType::SymbolicLink:
        RELEASE_ASSERT_NOT_REACHED();
    }

    return result;
}

Expected<Vector<String>, FileSystemStorageError> FileSystemStorageHandle::resolve(WebCore::FileSystemHandleIdentifier identifier)
{
    RefPtr manager = m_manager.get();
    if (!manager)
        return makeUnexpected(FileSystemStorageError::Unknown);

    auto path = manager->getPath(identifier);
    if (path.isEmpty())
        return makeUnexpected(FileSystemStorageError::Unknown);

    if (!path.startsWith(m_path))
        return Vector<String> { };

    auto restPath = path.substring(m_path.length());
    return restPath.split(pathSeparator);
}

Expected<FileSystemSyncAccessHandleInfo, FileSystemStorageError> FileSystemStorageHandle::createSyncAccessHandle()
{
    RefPtr manager = m_manager.get();
    if (!manager)
        return makeUnexpected(FileSystemStorageError::Unknown);

    bool acquired = manager->acquireLockForFile(m_path, identifier());
    if (!acquired)
        return makeUnexpected(FileSystemStorageError::InvalidState);

    auto handle = FileSystem::openFile(m_path, FileSystem::FileOpenMode::ReadWrite);
    if (handle == FileSystem::invalidPlatformFileHandle)
        return makeUnexpected(FileSystemStorageError::Unknown);

    auto ipcHandle = IPC::SharedFileHandle::create(std::exchange(handle, FileSystem::invalidPlatformFileHandle));
    if (!ipcHandle) {
        FileSystem::closeFile(handle);
        return makeUnexpected(FileSystemStorageError::BackendNotSupported);
    }

    ASSERT(!m_activeSyncAccessHandle);
    m_activeSyncAccessHandle = SyncAccessHandleInfo { WebCore::FileSystemSyncAccessHandleIdentifier::generate() };
    uint64_t initialCapacity = valueOrDefault(FileSystem::fileSize(m_path));
    return FileSystemSyncAccessHandleInfo { m_activeSyncAccessHandle->identifier, WTFMove(*ipcHandle), initialCapacity };
}

std::optional<FileSystemStorageError> FileSystemStorageHandle::closeSyncAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier accessHandleIdentifier)
{
    if (!m_activeSyncAccessHandle || m_activeSyncAccessHandle->identifier != accessHandleIdentifier)
        return FileSystemStorageError::Unknown;

    RefPtr manager = m_manager.get();
    if (!manager)
        return FileSystemStorageError::Unknown;

    manager->releaseLockForFile(m_path, identifier());
    m_activeSyncAccessHandle = std::nullopt;

    return std::nullopt;
}

std::optional<FileSystemStorageError> FileSystemStorageHandle::createWritable(bool keepExistingData)
{
    RefPtr manager = m_manager.get();
    if (!manager)
        return FileSystemStorageError::Unknown;

    bool acquired = manager->acquireLockForFile(m_path, identifier());
    if (!acquired)
        return FileSystemStorageError::InvalidState;

    auto path = FileSystem::createTemporaryFile("FileSystemWritableStream"_s);
    if (keepExistingData)
        FileSystem::copyFile(path, m_path);

    ASSERT(!m_activeWritableFile);
    m_activeWritableFile.open(path, FileSystem::FileOpenMode::ReadWrite);
    if (!m_activeWritableFile)
        return FileSystemStorageError::Unknown;

    return std::nullopt;
}

std::optional<FileSystemStorageError> FileSystemStorageHandle::closeWritable(WebCore::FileSystemWriteCloseReason reason)
{
    if (!m_activeWritableFile)
        return FileSystemStorageError::InvalidState;

    auto activeWritableFile = std::exchange(m_activeWritableFile, { });
    RefPtr manager = m_manager.get();
    if (!manager)
        return FileSystemStorageError::Unknown;

    manager->releaseLockForFile(m_path, identifier());

    if (reason == WebCore::FileSystemWriteCloseReason::Aborted) {
        m_activeWritableFile.close();
        FileSystem::deleteFile(m_activeWritableFile.path());
        return std::nullopt;
    }

    ASSERT(!activeWritableFile.path().isEmpty());
    if (FileSystem::copyFile(m_path, activeWritableFile.path()))
        return std::nullopt;

    return FileSystemStorageError::Unknown;
}

std::optional<FileSystemStorageError> FileSystemStorageHandle::executeCommandForWritableInternal(WebCore::FileSystemWriteCommandType type, std::optional<uint64_t> position, std::optional<uint64_t> size, std::span<const uint8_t> dataBytes, bool hasDataError)
{
    if (!m_activeWritableFile)
        return FileSystemStorageError::InvalidState;

    if (hasDataError)
        return FileSystemStorageError::InvalidDataType;

    switch (type) {
    case WebCore::FileSystemWriteCommandType::Write: {
        if (position) {
            auto result = FileSystem::seekFile(m_activeWritableFile.handle(), *position, FileSystem::FileSeekOrigin::Beginning);
            if (result == -1)
                return FileSystemStorageError::Unknown;
        }

        // FIXME: Add quota check.
        int result = FileSystem::writeToFile(m_activeWritableFile.handle(), dataBytes);
        if (result == -1)
            return FileSystemStorageError::Unknown;

        return std::nullopt;
    }
    case WebCore::FileSystemWriteCommandType::Seek: {
        if (!position)
            return FileSystemStorageError::MissingArgument;

        auto result = FileSystem::seekFile(m_activeWritableFile.handle(), *position, FileSystem::FileSeekOrigin::Beginning);
        if (result == -1)
            return FileSystemStorageError::Unknown;

        return std::nullopt;
    }
    case WebCore::FileSystemWriteCommandType::Truncate: {
        if (!size)
            return FileSystemStorageError::MissingArgument;

        bool truncated = FileSystem::truncateFile(m_activeWritableFile.handle(), *size);
        if (!truncated)
            return FileSystemStorageError::Unknown;

        FileSystem::seekFile(m_activeWritableFile.handle(), *size, FileSystem::FileSeekOrigin::Beginning);
        return std::nullopt;
    }
    }

    ASSERT_NOT_REACHED();
    return FileSystemStorageError::Unknown;
}

std::optional<FileSystemStorageError> FileSystemStorageHandle::executeCommandForWritable(WebCore::FileSystemWriteCommandType type, std::optional<uint64_t> position, std::optional<uint64_t> size, std::span<const uint8_t> dataBytes, bool hasDataError)
{
    auto error = executeCommandForWritableInternal(type, position, size, dataBytes, hasDataError);
    if (error)
        closeWritable(WebCore::FileSystemWriteCloseReason::Aborted);

    return error;
}

Expected<Vector<String>, FileSystemStorageError> FileSystemStorageHandle::getHandleNames()
{
    if (m_type != Type::Directory)
        return makeUnexpected(FileSystemStorageError::TypeMismatch);

    return FileSystem::listDirectory(m_path);
}

Expected<std::pair<WebCore::FileSystemHandleIdentifier, bool>, FileSystemStorageError> FileSystemStorageHandle::getHandle(IPC::Connection::UniqueID connection, String&& name)
{
    bool createIfNecessary = false;
    auto result = requestCreateHandle(connection, FileSystemStorageHandle::Type::Any, WTFMove(name), createIfNecessary);
    if (!result)
        return makeUnexpected(result.error());

    RefPtr manager = m_manager.get();
    if (!manager)
        return makeUnexpected(FileSystemStorageError::Unknown);

    auto resultType = manager->getType(result.value());
    ASSERT(resultType != FileSystemStorageHandle::Type::Any);
    return std::pair { result.value(), resultType == FileSystemStorageHandle::Type::Directory };
}

std::optional<FileSystemStorageError> FileSystemStorageHandle::move(WebCore::FileSystemHandleIdentifier destinationIdentifier, const String& newName)
{
    RefPtr manager = m_manager.get();
    if (!manager)
        return FileSystemStorageError::Unknown;

    // Do not move file if there is ongoing operation.
    if (m_activeSyncAccessHandle)
        return FileSystemStorageError::AccessHandleActive;

    if (manager->getType(destinationIdentifier) != Type::Directory)
        return FileSystemStorageError::TypeMismatch;

    auto path = manager->getPath(destinationIdentifier);
    if (path.isEmpty())
        return FileSystemStorageError::Unknown;

    if (!isValidFileName(path, newName))
        return FileSystemStorageError::InvalidName;

    auto destinationPath = FileSystem::pathByAppendingComponent(path, newName);
    if (!FileSystem::moveFile(m_path, destinationPath))
        return FileSystemStorageError::Unknown;

    m_path = destinationPath;
    m_name = newName;

    return std::nullopt;
}

std::optional<WebCore::FileSystemSyncAccessHandleIdentifier> FileSystemStorageHandle::activeSyncAccessHandle()
{
    if (!m_activeSyncAccessHandle)
        return std::nullopt;

    return m_activeSyncAccessHandle->identifier;
}

bool FileSystemStorageHandle::isActiveSyncAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier accessHandleIdentifier)
{
    return m_activeSyncAccessHandle && m_activeSyncAccessHandle->identifier == accessHandleIdentifier;
}

uint64_t FileSystemStorageHandle::allocatedUnusedCapacity()
{
    if (!m_activeSyncAccessHandle)
        return 0;

    auto actualSize = valueOrDefault(FileSystem::fileSize(m_path));
    return actualSize > m_activeSyncAccessHandle->capacity ? 0 : m_activeSyncAccessHandle->capacity - actualSize;
}

void FileSystemStorageHandle::requestNewCapacityForSyncAccessHandle(WebCore::FileSystemSyncAccessHandleIdentifier accessHandleIdentifier, uint64_t newCapacity, CompletionHandler<void(std::optional<uint64_t>)>&& completionHandler)
{
    if (!isActiveSyncAccessHandle(accessHandleIdentifier))
        return completionHandler(std::nullopt);

    uint64_t currentCapacity = m_activeSyncAccessHandle->capacity;
    if (newCapacity <= currentCapacity)
        return completionHandler(currentCapacity);

    RefPtr manager = m_manager.get();
    if (!manager)
        return completionHandler(std::nullopt);

    if (newCapacity < defaultInitialCapacity)
        newCapacity = defaultInitialCapacity;
    else if (newCapacity < defaultMaxCapacityForExponentialGrowth)
        newCapacity = pow(2, (int)std::log2(newCapacity) + 1);
    else
        newCapacity = defaultCapacityStep * ((newCapacity / defaultCapacityStep) + 1);

    manager->requestSpace(newCapacity - currentCapacity, [this, weakThis = WeakPtr { *this }, accessHandleIdentifier, newCapacity, completionHandler = WTFMove(completionHandler)](bool granted) mutable {
        if (!weakThis)
            return completionHandler(std::nullopt);

        if (!isActiveSyncAccessHandle(accessHandleIdentifier))
            return completionHandler(std::nullopt);

        if (granted)
            m_activeSyncAccessHandle->capacity = newCapacity;
        completionHandler(m_activeSyncAccessHandle->capacity);
    });
}

} // namespace WebKit
