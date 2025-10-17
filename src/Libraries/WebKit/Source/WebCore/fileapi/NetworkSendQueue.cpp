/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
#include "NetworkSendQueue.h"

#include "BlobLoader.h"
#include "ScriptExecutionContext.h"

namespace WebCore {

NetworkSendQueue::NetworkSendQueue(ScriptExecutionContext& context, WriteString&& writeString, WriteRawData&& writeRawData, ProcessError&& processError)
    : ContextDestructionObserver(&context)
    , m_writeString(WTFMove(writeString))
    , m_writeRawData(WTFMove(writeRawData))
    , m_processError(WTFMove(processError))
{
}

NetworkSendQueue::~NetworkSendQueue() = default;

void NetworkSendQueue::enqueue(CString&& utf8)
{
    if (m_queue.isEmpty()) {
        m_writeString(utf8);
        return;
    }
    m_queue.append(WTFMove(utf8));
}

void NetworkSendQueue::enqueue(const JSC::ArrayBuffer& binaryData, unsigned byteOffset, unsigned byteLength)
{
    if (m_queue.isEmpty()) {
        m_writeRawData(binaryData.span().subspan(byteOffset, byteLength));
        return;
    }
    m_queue.append(SharedBuffer::create(binaryData.span().subspan(byteOffset, byteLength)));
}

void NetworkSendQueue::enqueue(WebCore::Blob& blob)
{
    auto* context = scriptExecutionContext();
    if (!context)
        return;

    auto byteLength = blob.size();
    if (!byteLength) {
        // The cast looks weird, but is required for the overloading resolution to succeed.
        // Without it, there is an ambiguity where ArrayBuffer::create(const void* source, size_t byteLength) could be called instead.
        enqueue(JSC::ArrayBuffer::create(static_cast<size_t>(0U), 1), 0, 0);
        return;
    }
    auto blobLoader = makeUniqueRef<BlobLoader>([this](BlobLoader&) {
        processMessages();
    });
    auto* blobLoaderPtr = &blobLoader.get();
    m_queue.append(WTFMove(blobLoader));
    blobLoaderPtr->start(blob, context, FileReaderLoader::ReadAsArrayBuffer);
}

void NetworkSendQueue::clear()
{
    m_queue.clear();
}

void NetworkSendQueue::processMessages()
{
    while (!m_queue.isEmpty()) {
        bool shouldStopProcessing = false;
        switchOn(m_queue.first(), [this](const CString& utf8) {
            m_writeString(utf8);
        }, [this](Ref<FragmentedSharedBuffer>& data) {
            data->forEachSegment(m_writeRawData);
        }, [this, &shouldStopProcessing](UniqueRef<BlobLoader>& loader) {
            auto errorCode = loader->errorCode();
            if (loader->isLoading() || (errorCode && errorCode.value() == ExceptionCode::AbortError)) {
                shouldStopProcessing = true;
                return;
            }

            if (const auto& result = loader->arrayBufferResult()) {
                m_writeRawData(result->span());
                return;
            }
            ASSERT(errorCode);
            shouldStopProcessing = m_processError(errorCode.value()) == Continue::No;
        });
        if (shouldStopProcessing)
            return;
        m_queue.removeFirst();
    }

}

} // namespace WebCore
