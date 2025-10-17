/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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

#include "DOMFormData.h"
#include "ExceptionOr.h"
#include "FetchBodyConsumer.h"
#include "FormData.h"
#include "ReadableStream.h"
#include "URLSearchParams.h"
#include <variant>

namespace WebCore {

class DeferredPromise;
class FetchBodyOwner;
class FetchBodySource;
class ScriptExecutionContext;

class FetchBody {
public:
    void arrayBuffer(FetchBodyOwner&, Ref<DeferredPromise>&&);
    void blob(FetchBodyOwner&, Ref<DeferredPromise>&&);
    void bytes(FetchBodyOwner&, Ref<DeferredPromise>&&);
    void json(FetchBodyOwner&, Ref<DeferredPromise>&&);
    void text(FetchBodyOwner&, Ref<DeferredPromise>&&);
    void formData(FetchBodyOwner&, Ref<DeferredPromise>&&);

    void consumeAsStream(FetchBodyOwner&, FetchBodySource&);

    using Init = std::variant<RefPtr<Blob>, RefPtr<ArrayBufferView>, RefPtr<ArrayBuffer>, RefPtr<DOMFormData>, RefPtr<URLSearchParams>, RefPtr<ReadableStream>, String>;
    static ExceptionOr<FetchBody> extract(Init&&, String&);
    FetchBody() = default;
    FetchBody(FetchBody&&) = default;
    WEBCORE_EXPORT ~FetchBody();
    FetchBody& operator=(FetchBody&&) = default;

    explicit FetchBody(String&& data)
        : m_data(WTFMove(data))
    {
    }

    WEBCORE_EXPORT static std::optional<FetchBody> fromFormData(ScriptExecutionContext&, Ref<FormData>&&);

    void loadingFailed(const Exception&);
    void loadingSucceeded(const String& contentType);

    RefPtr<FormData> bodyAsFormData() const;

    using TakenData = std::variant<std::nullptr_t, Ref<FormData>, Ref<SharedBuffer>>;
    TakenData take();

    void setAsFormData(Ref<FormData>&& data) { m_data = WTFMove(data); }
    FetchBodyConsumer& consumer() { return m_consumer; }

    void consumeOnceLoadingFinished(FetchBodyConsumer::Type, Ref<DeferredPromise>&&);
    void cleanConsumer() { m_consumer.clean(); }
    bool hasConsumerPendingActivity() const { return m_consumer.hasPendingActivity(); }

    FetchBody clone();

    bool hasReadableStream() const { return !!m_readableStream; }
    const ReadableStream* readableStream() const { return m_readableStream.get(); }
    ReadableStream* readableStream() { return m_readableStream.get(); }
    void setReadableStream(Ref<ReadableStream>&& stream)
    {
        ASSERT(!m_readableStream);
        m_readableStream = WTFMove(stream);
    }

    void convertReadableStreamToArrayBuffer(FetchBodyOwner&, CompletionHandler<void(std::optional<Exception>&&)>&&);

    bool isBlob() const { return std::holds_alternative<Ref<const Blob>>(m_data); }
    bool isFormData() const { return std::holds_alternative<Ref<FormData>>(m_data); }
    bool isReadableStream() const { return std::holds_alternative<Ref<ReadableStream>>(m_data); }

private:
    explicit FetchBody(Ref<const Blob>&& data) : m_data(WTFMove(data)) { }
    explicit FetchBody(Ref<const ArrayBuffer>&& data) : m_data(WTFMove(data)) { }
    explicit FetchBody(Ref<const ArrayBufferView>&& data) : m_data(WTFMove(data)) { }
    explicit FetchBody(Ref<FormData>&& data) : m_data(WTFMove(data)) { }
    explicit FetchBody(Ref<const URLSearchParams>&& data) : m_data(WTFMove(data)) { }
    explicit FetchBody(Ref<ReadableStream>&& stream) : m_data(stream), m_readableStream(WTFMove(stream)) { }
    explicit FetchBody(FetchBodyConsumer&& consumer) : m_consumer(WTFMove(consumer)) { }

    void consume(FetchBodyOwner&, Ref<DeferredPromise>&&);

    void consumeArrayBuffer(FetchBodyOwner&, Ref<DeferredPromise>&&);
    void consumeArrayBufferView(FetchBodyOwner&, Ref<DeferredPromise>&&);
    void consumeText(FetchBodyOwner&, Ref<DeferredPromise>&&, const String&);
    void consumeBlob(FetchBodyOwner&, Ref<DeferredPromise>&&);
    void consumeFormData(FetchBodyOwner&, Ref<DeferredPromise>&&);

    bool isArrayBuffer() const { return std::holds_alternative<Ref<const ArrayBuffer>>(m_data); }
    bool isArrayBufferView() const { return std::holds_alternative<Ref<const ArrayBufferView>>(m_data); }
    bool isURLSearchParams() const { return std::holds_alternative<Ref<const URLSearchParams>>(m_data); }
    bool isText() const { return std::holds_alternative<String>(m_data); }

    const Blob& blobBody() const { return std::get<Ref<const Blob>>(m_data).get(); }
    FormData& formDataBody() { return std::get<Ref<FormData>>(m_data).get(); }
    const FormData& formDataBody() const { return std::get<Ref<FormData>>(m_data).get(); }
    const ArrayBuffer& arrayBufferBody() const { return std::get<Ref<const ArrayBuffer>>(m_data).get(); }
    const ArrayBufferView& arrayBufferViewBody() const { return std::get<Ref<const ArrayBufferView>>(m_data).get(); }
    String& textBody() { return std::get<String>(m_data); }
    const String& textBody() const { return std::get<String>(m_data); }
    const URLSearchParams& urlSearchParamsBody() const { return std::get<Ref<const URLSearchParams>>(m_data).get(); }

    using Data = std::variant<std::nullptr_t, Ref<const Blob>, Ref<FormData>, Ref<const ArrayBuffer>, Ref<const ArrayBufferView>, Ref<const URLSearchParams>, String, Ref<ReadableStream>>;
    Data m_data { nullptr };

    FetchBodyConsumer m_consumer { FetchBodyConsumer::Type::None };
    RefPtr<ReadableStream> m_readableStream;
};

struct FetchBodyWithType {
    FetchBody body;
    String type;
};

} // namespace WebCore
