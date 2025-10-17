/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
#include "BlobDataFileReference.h"

#include "File.h"
#include <wtf/FileSystem.h>

namespace WebCore {

BlobDataFileReference::BlobDataFileReference(const String& path, const String& replacementPath)
    : m_path(path)
    , m_replacementPath(replacementPath)
{
}

BlobDataFileReference::~BlobDataFileReference()
{
    if (!m_replacementPath.isNull())
        FileSystem::deleteFile(m_replacementPath);
}

const String& BlobDataFileReference::path()
{
#if ENABLE(FILE_REPLACEMENT)
    if (m_replacementShouldBeGenerated)
        generateReplacementFile();
#endif
    if (!m_replacementPath.isNull())
        return m_replacementPath;

    return m_path;
}

unsigned long long BlobDataFileReference::size()
{
#if ENABLE(FILE_REPLACEMENT)
    if (m_replacementShouldBeGenerated)
        generateReplacementFile();
#endif

    return m_size;
}

std::optional<WallTime> BlobDataFileReference::expectedModificationTime()
{
#if ENABLE(FILE_REPLACEMENT)
    // We do not currently track modifications for generated files, because we have a snapshot.
    // Unfortunately, this is inconsistent with regular file handling - File objects should be invalidated when underlying files change.
    if (m_replacementShouldBeGenerated || !m_replacementPath.isNull())
        return std::nullopt;
#endif
    return m_expectedModificationTime;
}

void BlobDataFileReference::startTrackingModifications()
{
    // This is not done automatically by the constructor, because BlobDataFileReference is
    // also used to pass paths around before registration. Only registered blobs need to pay
    // the cost of tracking file modifications.

#if ENABLE(FILE_REPLACEMENT)
    m_replacementShouldBeGenerated = File::shouldReplaceFile(m_path);
#endif

    // FIXME: Some platforms provide better ways to listen for file system object changes, consider using these.
    auto modificationTime = FileSystem::fileModificationTime(m_path);
    if (!modificationTime)
        return;

    m_expectedModificationTime = *modificationTime;

#if ENABLE(FILE_REPLACEMENT)
    if (m_replacementShouldBeGenerated)
        return;
#endif

    // This is a registered blob with a replacement file. Get the size of the replacement file.
    auto fileSize = FileSystem::fileSize(m_replacementPath.isNull() ? m_path : m_replacementPath);
    if (!fileSize)
        return;

    m_size = *fileSize;
}

void BlobDataFileReference::prepareForFileAccess()
{
}

void BlobDataFileReference::revokeFileAccess()
{
}

}
