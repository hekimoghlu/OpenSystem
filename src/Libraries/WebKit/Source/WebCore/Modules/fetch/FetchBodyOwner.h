/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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

#include "ActiveDOMObject.h"
#include "ExceptionOr.h"
#include "FetchBody.h"
#include "FetchBodySource.h"
#include "FetchHeaders.h"
#include "FetchLoader.h"
#include "FetchLoaderClient.h"
#include "ResourceError.h"
#include "SharedBuffer.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(FetchBodyOwner);

class FetchBodyOwner : public RefCountedAndCanMakeWeakPtr<FetchBodyOwner>, public ActiveDOMObject {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(FetchBodyOwner);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    ~FetchBodyOwner();

    bool bodyUsed() const { return isDisturbed(); }
    void arrayBuffer(Ref<DeferredPromise>&&);
    void blob(Ref<DeferredPromise>&&);
    void bytes(Ref<DeferredPromise>&&);
    void formData(Ref<DeferredPromise>&&);
    void json(Ref<DeferredPromise>&&);
    void text(Ref<DeferredPromise>&&);

    bool isDisturbed() const;
    bool isDisturbedOrLocked() const;

    void loadBlob(const Blob&, FetchBodyConsumer*);

    ExceptionOr<RefPtr<ReadableStream>> readableStream(JSC::JSGlobalObject&);
    bool hasReadableStreamBody() const { return m_body && m_body->hasReadableStream(); }
    bool isReadableStreamBody() const { return m_body && m_body->isReadableStream(); }

    virtual void consumeBodyAsStream();
    virtual void feedStream() { }
    virtual void cancel() { }
    virtual void loadBody() { }

    bool hasLoadingError() const;
    ResourceError loadingError() const;
    std::optional<Exception> loadingException() const;

    String contentType() const { return m_headers->fastGet(HTTPHeaderName::ContentType); }

    FetchBody& body() { return *m_body; }

protected:
    FetchBodyOwner(ScriptExecutionContext*, std::optional<FetchBody>&&, Ref<FetchHeaders>&&);

    const FetchBody& body() const { return *m_body; }
    bool isBodyNull() const { return !m_body; }
    bool isBodyNullOrOpaque() const { return !m_body || m_isBodyOpaque; }
    void cloneBody(FetchBodyOwner&);

    ExceptionOr<void> extractBody(FetchBody::Init&&);
    void consumeOnceLoadingFinished(FetchBodyConsumer::Type, Ref<DeferredPromise>&&);

    void setBody(FetchBody&& body) { m_body = WTFMove(body); }
    ExceptionOr<void> createReadableStream(JSC::JSGlobalObject&);

    // ActiveDOMObject.
    void stop() override;

    void setDisturbed() { m_isDisturbed = true; }

    void setBodyAsOpaque() { m_isBodyOpaque = true; }
    bool isBodyOpaque() const { return m_isBodyOpaque; }

    void setLoadingError(Exception&&);
    void setLoadingError(ResourceError&&);

private:
    // Blob loading routines
    void blobChunk(const SharedBuffer&);
    void blobLoadingSucceeded();
    void blobLoadingFailed();
    void finishBlobLoading();

    // ActiveDOMObject API
    bool virtualHasPendingActivity() const final;

    struct BlobLoader final : FetchLoaderClient {
        WTF_MAKE_TZONE_ALLOCATED(BlobLoader);
        WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(BlobLoader);
    public:
        BlobLoader(FetchBodyOwner&);

        // FetchLoaderClient API
        void didReceiveResponse(const ResourceResponse&) final;
        void didReceiveData(const SharedBuffer& buffer) final { owner.blobChunk(buffer); }
        void didFail(const ResourceError&) final;
        void didSucceed(const NetworkLoadMetrics&) final;

        FetchBodyOwner& owner;
        std::unique_ptr<FetchLoader> loader;
    };

protected:
    std::optional<FetchBody> m_body;
    bool m_isDisturbed { false };
    RefPtr<FetchBodySource> m_readableStreamSource;
    Ref<FetchHeaders> m_headers;

private:
    std::optional<BlobLoader> m_blobLoader;
    bool m_isBodyOpaque { false };

    std::variant<std::nullptr_t, Exception, ResourceError> m_loadingError;
};

} // namespace WebCore
