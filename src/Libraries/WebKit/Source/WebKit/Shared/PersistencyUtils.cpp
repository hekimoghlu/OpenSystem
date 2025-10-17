/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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
#include "PersistencyUtils.h"

#include "Logging.h"
#include <WebCore/SharedBuffer.h>
#include <wtf/FileSystem.h>
#include <wtf/RunLoop.h>

namespace WebKit {

using namespace WebCore;

std::unique_ptr<KeyedDecoder> createForFile(const String& path)
{
    ASSERT(!RunLoop::isMain());

    auto buffer = FileSystem::readEntireFile(path);
    if (!buffer)
        return nullptr;

    return KeyedDecoder::decoder(buffer->span());
}

void writeToDisk(std::unique_ptr<KeyedEncoder>&& encoder, String&& path)
{
    ASSERT(!RunLoop::isMain());

    auto rawData = encoder->finishEncoding();
    if (!rawData)
        return;

    FileSystem::PlatformFileHandle handle = FileSystem::openAndLockFile(path, FileSystem::FileOpenMode::Truncate);
    if (handle == FileSystem::invalidPlatformFileHandle)
        return;

    auto writtenBytes = FileSystem::writeToFile(handle, rawData->span());
    FileSystem::unlockAndCloseFile(handle);

    if (writtenBytes != static_cast<int64_t>(rawData->size()))
        RELEASE_LOG_ERROR(DiskPersistency, "Disk persistency: We only wrote %d out of %zu bytes to disk", static_cast<unsigned>(writtenBytes), rawData->size());
}

} // namespace WebKit
