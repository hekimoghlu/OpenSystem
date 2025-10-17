/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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

#include <wtf/FileSystem.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

// All methods are synchronous.
class FileStream {
    WTF_MAKE_TZONE_ALLOCATED(FileStream);
public:
    FileStream();
    ~FileStream();

    // Gets the size of a file. Also validates if the file has been changed or not if the expected modification time is provided, i.e. non-zero.
    // Returns total number of bytes if successful. -1 otherwise.
    long long getSize(const String& path, std::optional<WallTime> expectedModificationTime);

    // Opens a file for reading. The reading starts at the specified offset and lasts till the specified length.
    // Returns true on success. False otherwise.
    bool openForRead(const String& path, long long offset, long long length);

    // Closes the file.
    void close();

    // Reads a file into the provided data buffer.
    // Returns number of bytes being read on success. -1 otherwise.
    // If 0 is returned, it means that the reading is completed.
    int read(std::span<uint8_t> buffer);

private:
    FileSystem::PlatformFileHandle m_handle;
    long long m_bytesProcessed;
    long long m_totalBytesToRead;
};

} // namespace WebCore
