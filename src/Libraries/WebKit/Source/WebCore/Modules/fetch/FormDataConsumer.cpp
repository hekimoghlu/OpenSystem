/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 25, 2022.
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
#include "FormDataConsumer.h"

#include "BlobLoader.h"
#include "FormData.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/WorkQueue.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FormDataConsumer);

FormDataConsumer::FormDataConsumer(const FormData& formData, ScriptExecutionContext& context, Callback&& callback)
    : m_formData(formData.copy())
    , m_context(&context)
    , m_callback(WTFMove(callback))
    , m_fileQueue(WorkQueue::create("FormDataConsumer file queue"_s))
{
}

FormDataConsumer::~FormDataConsumer() = default;

void FormDataConsumer::read()
{
    if (isCancelled())
        return;

    ASSERT(m_callback);
    ASSERT(!m_blobLoader);

    if (m_currentElementIndex >= m_formData->elements().size()) {
        auto callback = std::exchange(m_callback, nullptr);
        callback(std::span<const uint8_t> { });
        return;
    }

    switchOn(m_formData->elements()[m_currentElementIndex++].data, [this](const Vector<uint8_t>& content) {
        consumeData(content);
    }, [this](const FormDataElement::EncodedFileData& fileData) {
        consumeFile(fileData.filename);
    }, [this](const FormDataElement::EncodedBlobData& blobData) {
        consumeBlob(blobData.url);
    });
}

void FormDataConsumer::consumeData(const Vector<uint8_t>& content)
{
    consume(content.span());
}

void FormDataConsumer::consumeFile(const String& filename)
{
    m_isReadingFile = true;
    m_fileQueue->dispatch([weakThis = WeakPtr { *this }, identifier = m_context->identifier(), path = filename.isolatedCopy()]() mutable {
        ScriptExecutionContext::postTaskTo(identifier, [weakThis = WTFMove(weakThis), content = FileSystem::readEntireFile(path)](auto&) {
            RefPtr protectedThis = weakThis.get();
            if (!protectedThis || !protectedThis->m_isReadingFile)
                return;

            protectedThis->m_isReadingFile = false;
            if (!content) {
                protectedThis->didFail(Exception { ExceptionCode::InvalidStateError, "Unable to read form data file"_s });
                return;
            }

            protectedThis->consume(*content);
        });
    });
}

void FormDataConsumer::consumeBlob(const URL& blobURL)
{
    m_blobLoader = makeUnique<BlobLoader>([weakThis = WeakPtr { *this }](BlobLoader&) mutable {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis)
            return;

        auto loader = std::exchange(protectedThis->m_blobLoader, { });
        if (!loader)
            return;

        if (auto optionalErrorCode = loader->errorCode()) {
            protectedThis->didFail(Exception { ExceptionCode::InvalidStateError, "Failed to read form data blob"_s });
            return;
        }

        if (auto data = loader->arrayBufferResult())
            protectedThis->consume(data->span());
    });

    m_blobLoader->start(blobURL, m_context.get(), FileReaderLoader::ReadAsArrayBuffer);

    if (!m_blobLoader || !m_blobLoader->isLoading())
        didFail(Exception { ExceptionCode::InvalidStateError, "Unable to read form data blob"_s });
}

void FormDataConsumer::consume(std::span<const uint8_t> content)
{
    if (!m_callback)
        return;

    if (!content.empty()) {
        bool result = m_callback(WTFMove(content));
        if (!result) {
            cancel();
            return;
        }

        if (!m_callback)
            return;
    }

    read();
}

void FormDataConsumer::didFail(Exception&& exception)
{
    auto callback = std::exchange(m_callback, nullptr);
    cancel();
    if (callback)
        callback(WTFMove(exception));
}

void FormDataConsumer::cancel()
{
    m_callback = nullptr;
    if (auto loader = std::exchange(m_blobLoader, { }))
        loader->cancel();
    m_isReadingFile = false;
    m_context = nullptr;
}

} // namespace WebCore
