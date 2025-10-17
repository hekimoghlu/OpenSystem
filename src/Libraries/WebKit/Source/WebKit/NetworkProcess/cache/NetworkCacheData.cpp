/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
#include "NetworkCacheData.h"

#include <fcntl.h>
#include <wtf/FileSystem.h>
#include <wtf/StdLibExtras.h>

#if !OS(WINDOWS)
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace WebKit {
namespace NetworkCache {

Data Data::mapToFile(const String& path) const
{
    FileSystem::PlatformFileHandle handle;
    auto applyData = [&](const Function<bool(std::span<const uint8_t>)>& applier) {
        apply(applier);
    };
    auto mappedFile = FileSystem::mapToFile(path, size(), WTFMove(applyData), &handle);
    if (!mappedFile)
        return { };
    return Data::adoptMap(WTFMove(mappedFile), handle);
}

Data mapFile(const String& path)
{
    auto file = FileSystem::openFile(path, FileSystem::FileOpenMode::Read);
    if (!FileSystem::isHandleValid(file))
        return { };
    auto size = FileSystem::fileSize(file);
    if (!size) {
        FileSystem::closeFile(file);
        return { };
    }
    return adoptAndMapFile(file, 0, *size);
}

Data adoptAndMapFile(FileSystem::PlatformFileHandle handle, size_t offset, size_t size)
{
    if (!size) {
        FileSystem::closeFile(handle);
        return Data::empty();
    }
    bool success;
    FileSystem::MappedFileData mappedFile(handle, FileSystem::FileOpenMode::Read, FileSystem::MappedFileMode::Private, success);
    if (!success) {
        FileSystem::closeFile(handle);
        return { };
    }

    return Data::adoptMap(WTFMove(mappedFile), handle);
}

SHA1::Digest computeSHA1(const Data& data, const Salt& salt)
{
    SHA1 sha1;
    sha1.addBytes(salt);
    data.apply([&sha1](std::span<const uint8_t> span) {
        sha1.addBytes(span);
        return true;
    });

    SHA1::Digest digest;
    sha1.computeHash(digest);
    return digest;
}

bool bytesEqual(const Data& a, const Data& b)
{
    if (a.isNull() || b.isNull())
        return false;
    return equalSpans(a.span(), b.span());
}

} // namespace NetworkCache
} // namespace WebKit
