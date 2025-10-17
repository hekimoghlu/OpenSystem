/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

#include <wtf/Assertions.h>
#include <wtf/FileSystem.h>

namespace WebCore {

class WEBCORE_EXPORT FileHandle final {
public:
    FileHandle() = default;
    ~FileHandle();
    FileHandle(const String& path, FileSystem::FileOpenMode);
    FileHandle(const String& path, FileSystem::FileOpenMode, OptionSet<FileSystem::FileLockMode>);
    FileHandle(FileHandle&& other);
    FileHandle& operator=(FileHandle&& other);
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    explicit FileHandle(FileSystem::PlatformFileHandle);

    explicit operator bool() const;
    String path() const;

    bool open(const String& path, FileSystem::FileOpenMode);
    bool open();
    int read(std::span<uint8_t> data);
    int write(std::span<const uint8_t> data);
    bool printf(const char* format, ...) WTF_ATTRIBUTE_PRINTF(2, 3);
    void close();

    FileSystem::PlatformFileHandle handle() const;

    FileHandle isolatedCopy() && { return WTFMove(*this); }

private:
    String m_path;
    FileSystem::FileOpenMode m_mode { FileSystem::FileOpenMode::Read };
    FileSystem::PlatformFileHandle m_fileHandle { FileSystem::invalidPlatformFileHandle };
    OptionSet<FileSystem::FileLockMode> m_lockMode;
    bool m_shouldLock { false };
};

} // namespace WebCore
