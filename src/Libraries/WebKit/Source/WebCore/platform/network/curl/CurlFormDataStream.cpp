/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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
#include "CurlFormDataStream.h"

#if USE(CURL)

#include "BlobRegistry.h"
#include "Logging.h"
#include <wtf/MainThread.h>

namespace WebCore {

CurlFormDataStream::CurlFormDataStream(const RefPtr<FormData>& formData)
{
    ASSERT(isMainThread());

    if (!formData || formData->isEmpty())
        return;

    m_formData = formData->isolatedCopy();

    // Resolve the blob elements so the formData can correctly report it's size.
    m_formData = m_formData->resolveBlobReferences(blobRegistry().blobRegistryImpl());
}

CurlFormDataStream::~CurlFormDataStream()
{

}

void CurlFormDataStream::clean()
{
    if (m_postData)
        m_postData = nullptr;

    if (m_fileHandle != FileSystem::invalidPlatformFileHandle) {
        FileSystem::closeFile(m_fileHandle);
        m_fileHandle = FileSystem::invalidPlatformFileHandle;
    }
}

const Vector<uint8_t>* CurlFormDataStream::getPostData()
{
    if (!m_formData)
        return nullptr;

    if (!m_postData)
        m_postData = makeUnique<Vector<uint8_t>>(m_formData->flatten());

    return m_postData.get();
}

unsigned long long CurlFormDataStream::totalSize()
{
    if (!m_formData)
        return 0;

    if (m_isContentLengthUpdated)
        return m_totalSize;

    m_isContentLengthUpdated = true;

    for (const auto& element : m_formData->elements())
        m_totalSize += element.lengthInBytes();

    return m_totalSize;
}

std::optional<size_t> CurlFormDataStream::read(char* buffer, size_t size)
{
    if (!m_formData)
        return std::nullopt;

    const auto totalElementSize = m_formData->elements().size();
    if (m_elementPosition >= totalElementSize)
        return 0;

    size_t totalReadBytes = 0;

    while ((m_elementPosition < totalElementSize) && (totalReadBytes < size)) {
        const auto& element = m_formData->elements().at(m_elementPosition);

        size_t bufferSize = size - totalReadBytes;
        char* bufferPosition = buffer + totalReadBytes;

        std::optional<size_t> readBytes = switchOn(element.data,
            [&] (const Vector<uint8_t>& bytes) {
                return readFromData(bytes, bufferPosition, bufferSize);
            }, [&] (const FormDataElement::EncodedFileData& fileData) {
                return readFromFile(fileData, bufferPosition, bufferSize);
            }, [] (const FormDataElement::EncodedBlobData&) -> std::optional<size_t> {
                ASSERT_NOT_REACHED();
                return std::nullopt;
            }
        );

        if (!readBytes)
            return std::nullopt;

        totalReadBytes += *readBytes;
    }

    m_totalReadSize += totalReadBytes;

    return totalReadBytes;
}

std::optional<size_t> CurlFormDataStream::readFromFile(const FormDataElement::EncodedFileData& fileData, char* buffer, size_t size)
{
    if (m_fileHandle == FileSystem::invalidPlatformFileHandle)
        m_fileHandle = FileSystem::openFile(fileData.filename, FileSystem::FileOpenMode::Read);

    if (!FileSystem::isHandleValid(m_fileHandle)) {
        LOG(Network, "Curl - Failed while trying to open %s for upload\n", fileData.filename.utf8().data());
        m_fileHandle = FileSystem::invalidPlatformFileHandle;
        return std::nullopt;
    }

    auto readBytes = FileSystem::readFromFile(m_fileHandle, { byteCast<uint8_t>(buffer), size });
    if (readBytes < 0) {
        LOG(Network, "Curl - Failed while trying to read %s for upload\n", fileData.filename.utf8().data());
        FileSystem::closeFile(m_fileHandle);
        m_fileHandle = FileSystem::invalidPlatformFileHandle;
        return std::nullopt;
    }

    if (!readBytes) {
        FileSystem::closeFile(m_fileHandle);
        m_fileHandle = FileSystem::invalidPlatformFileHandle;
        m_elementPosition++;
    }

    return readBytes;
}

std::optional<size_t> CurlFormDataStream::readFromData(const Vector<uint8_t>& data, char* buffer, size_t size)
{
    size_t elementSize = data.size() - m_dataOffset;
    const uint8_t* elementBuffer = data.data() + m_dataOffset;

    size_t readBytes = elementSize > size ? size : elementSize;
    memcpy(buffer, elementBuffer, readBytes);

    if (elementSize > readBytes)
        m_dataOffset += readBytes;
    else {
        m_dataOffset = 0;
        m_elementPosition++;
    }

    return readBytes;
}

} // namespace WebCore

#endif
