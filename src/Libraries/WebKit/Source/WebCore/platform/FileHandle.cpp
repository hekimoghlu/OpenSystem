/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 7, 2025.
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
#include "FileHandle.h"

namespace WebCore {

FileHandle::FileHandle(const String& path, FileSystem::FileOpenMode mode)
    : m_path { path }
    , m_mode { mode }
{
}

FileHandle::FileHandle(FileHandle&& other)
    : m_path { WTFMove(other.m_path) }
    , m_mode { WTFMove(other.m_mode) }
    , m_fileHandle { std::exchange(other.m_fileHandle, FileSystem::invalidPlatformFileHandle) }
{
}

FileHandle::FileHandle(const String& path, FileSystem::FileOpenMode mode, OptionSet<FileSystem::FileLockMode> lockMode)
    : m_path { path }
    , m_mode { mode }
    , m_lockMode { lockMode }
    , m_shouldLock { true }
{
}

FileHandle::FileHandle(FileSystem::PlatformFileHandle handle)
    : m_fileHandle(handle)
{
}

FileHandle::~FileHandle()
{
    close();
}

FileHandle& FileHandle::operator=(FileHandle&& other)
{
    close();
    m_path = WTFMove(other.m_path);
    m_mode = WTFMove(other.m_mode);
    m_fileHandle = std::exchange(other.m_fileHandle, FileSystem::invalidPlatformFileHandle);
    m_shouldLock = other.m_shouldLock;
    m_lockMode = other.m_lockMode;

    return *this;
}

FileHandle::operator bool() const
{
    return FileSystem::isHandleValid(m_fileHandle);
}

bool FileHandle::open(const String& path, FileSystem::FileOpenMode mode)
{
    if (*this && path == m_path && mode == m_mode)
        return true;

    close();
    m_path = path;
    m_mode = mode;
    return open();
}

bool FileHandle::open()
{
    if (m_path.isEmpty())
        return false;

    if (!*this)
        m_fileHandle = m_shouldLock ? FileSystem::openAndLockFile(m_path, m_mode, m_lockMode) :  FileSystem::openFile(m_path, m_mode);

    return static_cast<bool>(*this);
}

int FileHandle::read(std::span<uint8_t> data)
{
    if (!open())
        return -1;

    return FileSystem::readFromFile(m_fileHandle, data);
}

int FileHandle::write(std::span<const uint8_t> data)
{
    if (!open())
        return -1;

    return FileSystem::writeToFile(m_fileHandle, data);
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

bool FileHandle::printf(const char* format, ...)
{
    va_list args;
    va_start(args, format);

    va_list preflightArgs;
    va_copy(preflightArgs, args);
    size_t stringLength = vsnprintf(nullptr, 0, format, preflightArgs);
    va_end(preflightArgs);

    Vector<char, 1024> buffer(stringLength + 1);
    vsnprintf(buffer.data(), stringLength + 1, format, args);

    va_end(args);

    return write(byteCast<uint8_t>(buffer.mutableSpan()).first(stringLength)) >= 0;
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

void FileHandle::close()
{
    if (m_shouldLock && *this) {
        // FileSystem::unlockAndCloseFile requires the file handle to be valid while closeFile does not
        FileSystem::unlockAndCloseFile(m_fileHandle);
        return;
    }

    FileSystem::closeFile(m_fileHandle);
}

FileSystem::PlatformFileHandle FileHandle::handle() const
{
    return m_fileHandle;
}

String FileHandle::path() const
{
    return m_path;
}

} // namespace WebCore
