/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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
#include "FileStream.h"

#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FileStream);

FileStream::FileStream()
    : m_handle(FileSystem::invalidPlatformFileHandle)
    , m_bytesProcessed(0)
    , m_totalBytesToRead(0)
{
}

FileStream::~FileStream()
{
    close();
}

long long FileStream::getSize(const String& path, std::optional<WallTime> expectedModificationTime)
{
    // Check the modification time for the possible file change.
    auto modificationTime = FileSystem::fileModificationTime(path);
    if (!modificationTime)
        return -1;
    if (expectedModificationTime) {
        if (expectedModificationTime->secondsSinceEpoch().secondsAs<time_t>() != modificationTime->secondsSinceEpoch().secondsAs<time_t>())
            return -1;
    }

    // Now get the file size.
    auto length = FileSystem::fileSize(path);
    if (!length)
        return -1;

    return *length;
}

bool FileStream::openForRead(const String& path, long long offset, long long length)
{
    if (FileSystem::isHandleValid(m_handle))
        return true;

    // Open the file.
    m_handle = FileSystem::openFile(path, FileSystem::FileOpenMode::Read);
    if (!FileSystem::isHandleValid(m_handle))
        return false;

    // Jump to the beginning position if the file has been sliced.
    if (offset > 0) {
        if (FileSystem::seekFile(m_handle, offset, FileSystem::FileSeekOrigin::Beginning) < 0)
            return false;
    }

    m_totalBytesToRead = length;
    m_bytesProcessed = 0;

    return true;
}

void FileStream::close()
{
    if (FileSystem::isHandleValid(m_handle)) {
        FileSystem::closeFile(m_handle);
        m_handle = FileSystem::invalidPlatformFileHandle;
    }
}

int FileStream::read(std::span<uint8_t> buffer)
{
    if (!FileSystem::isHandleValid(m_handle))
        return -1;

    long long remaining = m_totalBytesToRead - m_bytesProcessed;
    int bytesToRead = remaining < static_cast<int>(buffer.size()) ? static_cast<int>(remaining) : static_cast<int>(buffer.size());
    int bytesRead = 0;
    if (bytesToRead > 0)
        bytesRead = FileSystem::readFromFile(m_handle, buffer.first(bytesToRead));
    if (bytesRead < 0)
        return -1;
    if (bytesRead > 0)
        m_bytesProcessed += bytesRead;

    return bytesRead;
}

} // namespace WebCore
