/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#include "NetworkCacheBlobStorage.h"

#include "Logging.h"
#include "NetworkCacheFileSystem.h"
#include <fcntl.h>
#include <wtf/FileSystem.h>
#include <wtf/RunLoop.h>
#include <wtf/SHA1.h>

#if !OS(WINDOWS)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace WebKit {
namespace NetworkCache {

BlobStorage::BlobStorage(const String& blobDirectoryPath, Salt salt)
    : m_blobDirectoryPath(crossThreadCopy(blobDirectoryPath))
    , m_salt(salt)
{
}

String BlobStorage::blobDirectoryPathIsolatedCopy() const
{
    return m_blobDirectoryPath.isolatedCopy();
}

void BlobStorage::synchronize()
{
    ASSERT(!RunLoop::isMain());

    auto blobDirectoryPath = blobDirectoryPathIsolatedCopy();
    FileSystem::makeAllDirectories(blobDirectoryPath);

    m_approximateSize = 0;
    auto blobDirectory = blobDirectoryPath;
    traverseDirectory(blobDirectory, [this, &blobDirectory](const String& name, DirectoryEntryType type) {
        if (type != DirectoryEntryType::File)
            return;
        auto path = FileSystem::pathByAppendingComponent(blobDirectory, name);
        auto linkCount = FileSystem::hardLinkCount(path);
        // No clients left for this blob.
        if (linkCount && *linkCount == 1)
            FileSystem::deleteFile(path);
        else
            m_approximateSize += FileSystem::fileSize(path).value_or(0);
    });

    LOG(NetworkCacheStorage, "(NetworkProcess) blob synchronization completed approximateSize=%zu", approximateSize());
}

String BlobStorage::blobPathForHash(const SHA1::Digest& hash) const
{
    auto hashAsString = SHA1::hexDigest(hash);
    return FileSystem::pathByAppendingComponent(blobDirectoryPathIsolatedCopy(), StringView::fromLatin1(hashAsString.data()));
}

BlobStorage::Blob BlobStorage::add(const String& path, const Data& data)
{
    ASSERT(!RunLoop::isMain());

    auto hash = computeSHA1(data, m_salt);
    if (data.isEmpty())
        return { data, hash };

    String blobPath = blobPathForHash(hash);
    
    FileSystem::deleteFile(path);

    bool blobExists = FileSystem::fileExists(blobPath);
    if (blobExists) {
        if (FileSystem::makeSafeToUseMemoryMapForPath(blobPath)) {
            auto existingData = mapFile(blobPath);
            if (bytesEqual(existingData, data)) {
                if (!FileSystem::hardLink(blobPath, path))
                    WTFLogAlways("Failed to create hard link from %s to %s", blobPath.utf8().data(), path.utf8().data());
                return { existingData, hash };
            }
        }
        FileSystem::deleteFile(blobPath);
    }

    auto mappedData = data.mapToFile(blobPath);
    if (mappedData.isNull())
        return { };

    if (!FileSystem::hardLink(blobPath, path))
        WTFLogAlways("Failed to create hard link from %s to %s", blobPath.utf8().data(), path.utf8().data());

    m_approximateSize += mappedData.size();

    return { mappedData, hash };
}

BlobStorage::Blob BlobStorage::get(const String& path)
{
    ASSERT(!RunLoop::isMain());

    auto data = mapFile(path);

    return { data, computeSHA1(data, m_salt) };
}

void BlobStorage::remove(const String& path)
{
    ASSERT(!RunLoop::isMain());

    FileSystem::deleteFile(path);
}

unsigned BlobStorage::shareCount(const String& path)
{
    ASSERT(!RunLoop::isMain());

    auto linkCount = FileSystem::hardLinkCount(path);
    if (!linkCount)
        return 0;
    // Link count is 2 in the single client case (the blob file and a link).
    return *linkCount - 1;
}

}
}
