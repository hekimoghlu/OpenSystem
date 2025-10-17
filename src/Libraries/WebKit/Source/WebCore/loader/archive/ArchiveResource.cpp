/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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
#include "ArchiveResource.h"

#include "SharedBuffer.h"
#include <wtf/RunLoop.h>

namespace WebCore {

inline ArchiveResource::ArchiveResource(Ref<FragmentedSharedBuffer>&& data, const URL& url, const String& mimeType, const String& textEncoding, const String& frameName, const ResourceResponse& response, const String& relativeFilePath)
    : SubstituteResource(URL { url }, ResourceResponse { response }, WTFMove(data))
    , m_mimeType(mimeType)
    , m_textEncoding(textEncoding)
    , m_frameName(frameName)
    , m_relativeFilePath(relativeFilePath)
    , m_shouldIgnoreWhenUnarchiving(false)
{
}

RefPtr<ArchiveResource> ArchiveResource::create(RefPtr<FragmentedSharedBuffer>&& data, const URL& url, const String& mimeType, const String& textEncoding, const String& frameName, const ResourceResponse& response, const String& relativeFilePath)
{
    if (!data)
        return nullptr;
    if (response.isNull()) {
        ResourceResponse syntheticResponse(url, mimeType, data->size(), textEncoding);
        // Provide a valid HTTP status code for http URLs since we have logic in WebCore that validates it.
        if (url.protocolIsInHTTPFamily())
            syntheticResponse.setHTTPStatusCode(200);
        return adoptRef(*new ArchiveResource(data.releaseNonNull(), url, mimeType, textEncoding, frameName, WTFMove(syntheticResponse), relativeFilePath));
    }
    return adoptRef(*new ArchiveResource(data.releaseNonNull(), url, mimeType, textEncoding, frameName, response, relativeFilePath));
}

RefPtr<ArchiveResource> ArchiveResource::create(RefPtr<FragmentedSharedBuffer>&& data, const URL& url, const ResourceResponse& response)
{
    return create(WTFMove(data), url, response.mimeType(), response.textEncodingName(), String(), response);
}

Expected<String, ArchiveError> ArchiveResource::saveToDisk(const String& directory)
{
    ASSERT(!RunLoop::isMain());

    if (directory.isEmpty() || m_relativeFilePath.isEmpty())
        return makeUnexpected(ArchiveError::InvalidFilePath);

    auto filePath = FileSystem::pathByAppendingComponent(directory, m_relativeFilePath);
    FileSystem::makeAllDirectories(FileSystem::parentPath(filePath));
    auto fileData = data().extractData();
    int bytesWritten = FileSystem::overwriteEntireFile(filePath, fileData.span());

    if (bytesWritten < 0)
        return makeUnexpected(ArchiveError::FileSystemError);

    if ((size_t)bytesWritten != fileData.size()) {
        FileSystem::deleteFile(filePath);
        return makeUnexpected(ArchiveError::FileSystemError);
    }

    return filePath;
}

}
