/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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
#include "ShareDataReader.h"

#include "BlobLoader.h"
#include "Document.h"
#include "SharedBuffer.h"

namespace WebCore {

ShareDataReader::ShareDataReader(CompletionHandler<void(ExceptionOr<ShareDataWithParsedURL&>)>&& completionHandler)
    : m_completionHandler(WTFMove(completionHandler))
{

}

ShareDataReader::~ShareDataReader()
{
    cancel();
}

void ShareDataReader::start(Document* document, ShareDataWithParsedURL&& shareData)
{
    m_filesReadSoFar = 0;
    m_shareData = WTFMove(shareData);
    int count = 0;
    m_pendingFileLoads.reserveInitialCapacity(m_shareData.shareData.files.size());
    for (auto& blob : m_shareData.shareData.files) {
        m_pendingFileLoads.append(makeUniqueRef<BlobLoader>([this, count, fileName = blob->name()](BlobLoader&) {
            this->didFinishLoading(count, fileName);
        }));
        m_pendingFileLoads.last()->start(blob, document, FileReaderLoader::ReadAsArrayBuffer);
        if (m_pendingFileLoads.isEmpty()) {
            // The previous load failed synchronously and cancel() was called. We should not attempt to do any further loads.
            break;
        }
        ++count;
    }
}

void ShareDataReader::didFinishLoading(int loadIndex, const String& fileName)
{
    if (m_pendingFileLoads.isEmpty()) {
        // cancel() was called.
        return;
    }

    if (m_pendingFileLoads[loadIndex]->errorCode()) {
        if (auto completionHandler = std::exchange(m_completionHandler, { }))
            completionHandler(Exception { ExceptionCode::AbortError, "Abort due to error while reading files."_s });
        cancel();
        return;
    }

    auto arrayBuffer = m_pendingFileLoads[loadIndex]->arrayBufferResult();

    RawFile file;
    file.fileName = fileName;
    file.fileData = SharedBuffer::create(arrayBuffer->span());
    m_shareData.files.append(WTFMove(file));
    m_filesReadSoFar++;

    if (m_filesReadSoFar == static_cast<int>(m_pendingFileLoads.size())) {
        m_pendingFileLoads.clear();
        if (auto completionHandler = std::exchange(m_completionHandler, { }))
            completionHandler(m_shareData);
    }
}

void ShareDataReader::cancel()
{
    m_pendingFileLoads.clear();
}

}
