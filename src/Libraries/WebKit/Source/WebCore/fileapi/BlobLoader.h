/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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
#pragma once

#include "Blob.h"
#include "Document.h"
#include "ExceptionCode.h"
#include "FileReaderLoader.h"
#include "FileReaderLoaderClient.h"
#include "Logging.h"
#include "SharedBuffer.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <wtf/CompletionHandler.h>

namespace WebCore {

class BlobLoader final : public FileReaderLoaderClient {
    WTF_MAKE_TZONE_ALLOCATED(BlobLoader);
public:
    // CompleteCallback is always called except if BlobLoader is cancelled/deallocated.
    using CompleteCallback = Function<void(BlobLoader&)>;
    explicit BlobLoader(CompleteCallback&&);
    ~BlobLoader();

    void start(Blob&, ScriptExecutionContext*, FileReaderLoader::ReadType);
    void start(const URL&, ScriptExecutionContext*, FileReaderLoader::ReadType);

    void cancel();
    bool isLoading() const { return m_loader && m_completeCallback; }
    String stringResult() const { return m_loader ? m_loader->stringResult() : String(); }
    RefPtr<JSC::ArrayBuffer> arrayBufferResult() const { return m_loader ? m_loader->arrayBufferResult() : nullptr; }
    std::optional<ExceptionCode> errorCode() const { return m_loader ? m_loader->errorCode() : std::nullopt; }

private:
    void didStartLoading() final { }
    void didReceiveData() final { }

    void didFinishLoading() final;
    void didFail(ExceptionCode errorCode) final;
    void complete();

    std::unique_ptr<FileReaderLoader> m_loader;
    CompleteCallback m_completeCallback;
};

inline BlobLoader::BlobLoader(CompleteCallback&& completeCallback)
    : m_completeCallback(WTFMove(completeCallback))
{
}

inline BlobLoader::~BlobLoader()
{
    if (isLoading())
        cancel();
}

inline void BlobLoader::cancel()
{
    RELEASE_LOG_INFO_IF(m_completeCallback, Loading, "Cancelling ongoing blob loader");
    if (m_loader)
        m_loader->cancel();
}

inline void BlobLoader::start(Blob& blob, ScriptExecutionContext* context, FileReaderLoader::ReadType readType)
{
    ASSERT(!m_loader);
    m_loader = makeUnique<FileReaderLoader>(readType, this);
    m_loader->start(context, blob);
}

inline void BlobLoader::start(const URL& blobURL, ScriptExecutionContext* context, FileReaderLoader::ReadType readType)
{
    ASSERT(!m_loader);
    m_loader = makeUnique<FileReaderLoader>(readType, this);
    m_loader->start(context, blobURL);
}

inline void BlobLoader::didFinishLoading()
{
    std::exchange(m_completeCallback, { })(*this);
}

inline void BlobLoader::didFail(ExceptionCode)
{
    std::exchange(m_completeCallback, { })(*this);
}

} // namespace WebCore
