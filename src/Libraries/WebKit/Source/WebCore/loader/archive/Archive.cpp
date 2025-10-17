/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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
#include "Archive.h"

#include <wtf/RunLoop.h>
#include <wtf/Scope.h>

namespace WebCore {

Archive::~Archive() = default;

void Archive::clearAllSubframeArchives()
{
    UncheckedKeyHashSet<Archive*> clearedArchives;
    clearedArchives.add(this);
    clearAllSubframeArchives(clearedArchives);
}

void Archive::clearAllSubframeArchives(UncheckedKeyHashSet<Archive*>& clearedArchives)
{
    ASSERT(clearedArchives.contains(this));
    for (auto& archive : m_subframeArchives) {
        if (clearedArchives.add(archive.ptr()))
            archive->clearAllSubframeArchives(clearedArchives);
    }
    m_subframeArchives.clear();
}

Expected<Vector<String>, ArchiveError> Archive::saveResourcesToDisk(const String& directory)
{
    ASSERT(!RunLoop::isMain());

    Vector<String> filePaths;
    if (!m_mainResource)
        return makeUnexpected(ArchiveError::EmptyResource);

    bool hasError = false;
    auto cleanup = makeScopeExit([&] {
        if (hasError) {
            for (auto filePath : filePaths)
                FileSystem::deleteFile(filePath);
        }
    });

    auto mainResourceResult = m_mainResource->saveToDisk(directory);
    if (!mainResourceResult) {
        hasError = true;
        return makeUnexpected(mainResourceResult.error());
    }
    filePaths.append(mainResourceResult.value());

    for (auto subresource : m_subresources) {
        auto result = subresource->saveToDisk(directory);
        if (!result) {
            hasError = true;
            return makeUnexpected(result.error());
        }
        filePaths.append(result.value());
    }

    for (auto subframeArchive : m_subframeArchives) {
        auto result = subframeArchive->saveResourcesToDisk(directory);
        if (!result) {
            hasError = true;
            return makeUnexpected(result.error());
        }
        filePaths.appendVector(result.value());
    }

    return filePaths;
}

}
