/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
#import "config.h"
#import "BlobDataFileReference.h"

#if ENABLE(FILE_REPLACEMENT)

#import <wtf/FileSystem.h>
#import <wtf/text/CString.h>

namespace WebCore {

void BlobDataFileReference::generateReplacementFile()
{
    ASSERT(m_replacementPath.isNull());
    ASSERT(m_replacementShouldBeGenerated);

    prepareForFileAccess();

    auto generatedFile = FileSystem::createTemporaryZipArchive(m_path);
    if (!generatedFile.isNull())
        m_replacementPath = WTFMove(generatedFile);

    m_replacementShouldBeGenerated = false;
    if (!m_replacementPath.isNull()) {
        if (auto fileSize = FileSystem::fileSize(m_replacementPath))
            m_size = *fileSize;
    }

    revokeFileAccess();
}

}

#endif
