/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
#include "FetchBody.h"

#include "Document.h"
#include "FetchBodyOwner.h"
#include "FetchBodySource.h"
#include "FetchHeaders.h"
#include "HTTPHeaderValues.h"
#include "HTTPParsers.h"
#include "JSDOMFormData.h"
#include "JSDOMPromiseDeferred.h"
#include "ReadableStreamSource.h"
#include <JavaScriptCore/ArrayBufferView.h>
#include <pal/text/TextCodecUTF8.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

FetchBody::~FetchBody() = default;

ExceptionOr<FetchBody> FetchBody::extract(Init&& value, String& contentType)
{
    return WTF::switchOn(value, [&](RefPtr<Blob>& value) mutable -> ExceptionOr<FetchBody> {
        Ref<const Blob> blob = value.releaseNonNull();
        if (!blob->type().isEmpty())
            contentType = blob->type();
        return FetchBody(WTFMove(blob));
    }, [&](RefPtr<DOMFormData>& value) mutable -> ExceptionOr<FetchBody> {
        Ref<DOMFormData> domFormData = value.releaseNonNull();
        auto formData = FormData::createMultiPart(domFormData.get());
        contentType = makeString("multipart/form-data; boundary="_s, formData->boundary());
        return FetchBody(WTFMove(formData));
    }, [&](RefPtr<URLSearchParams>& value) mutable -> ExceptionOr<FetchBody> {
        Ref<const URLSearchParams> params = value.releaseNonNull();
        contentType = HTTPHeaderValues::formURLEncodedContentType();
        return FetchBody(WTFMove(params));
    }, [&](RefPtr<ArrayBuffer>& value) mutable -> ExceptionOr<FetchBody> {
        Ref<const ArrayBuffer> buffer = value.releaseNonNull();
        return FetchBody(WTFMove(buffer));
    }, [&](RefPtr<ArrayBufferView>& value) mutable -> ExceptionOr<FetchBody> {
        Ref<const ArrayBufferView> buffer = value.releaseNonNull();
        return FetchBody(WTFMove(buffer));
    }, [&](RefPtr<ReadableStream>& stream) mutable -> ExceptionOr<FetchBody> {
        if (stream->isDisturbed())
            return Exception { ExceptionCode::TypeError, "Input body is disturbed."_s };
        if (stream->isLocked())
            return Exception { ExceptionCode::TypeError, "Input body is locked."_s };

        return FetchBody(stream.releaseNonNull());
    }, [&](String& value) -> ExceptionOr<FetchBody> {
        contentType = HTTPHeaderValues::textPlainContentType();
        return FetchBody(WTFMove(value));
    });
}

std::optional<FetchBody> FetchBody::fromFormData(ScriptExecutionContext& context, Ref<FormData>&& formData)
{
    ASSERT(!formData->isEmpty());

    if (auto buffer = formData->asSharedBuffer()) {
        FetchBody body;
        body.m_consumer.setData(buffer.releaseNonNull());
        return body;
    }

    auto url = formData->asBlobURL();
    if (!url.isNull()) {
        // FIXME: Properly set mime type and size of the blob.
        Ref<const Blob> blob = Blob::deserialize(&context, url, { }, { }, 0, { });
        return FetchBody { WTFMove(blob) };
    }

    return FetchBody { WTFMove(formData) };
}

void FetchBody::arrayBuffer(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise)
{
    m_consumer.setType(FetchBodyConsumer::Type::ArrayBuffer);
    consume(owner, WTFMove(promise));
}

void FetchBody::blob(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise)
{
    m_consumer.setType(FetchBodyConsumer::Type::Blob);
    consume(owner, WTFMove(promise));
}

void FetchBody::bytes(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise)
{
    m_consumer.setType(FetchBodyConsumer::Type::Bytes);
    consume(owner, WTFMove(promise));
}

void FetchBody::json(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise)
{
    if (isText()) {
        fulfillPromiseWithJSON(WTFMove(promise), textBody());
        return;
    }
    m_consumer.setType(FetchBodyConsumer::Type::JSON);
    consume(owner, WTFMove(promise));
}

void FetchBody::text(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise)
{
    if (isText()) {
        promise->resolve<IDLDOMString>(textBody());
        return;
    }
    m_consumer.setType(FetchBodyConsumer::Type::Text);
    consume(owner, WTFMove(promise));
}

void FetchBody::formData(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise)
{
    m_consumer.setType(FetchBodyConsumer::Type::FormData);
    consume(owner, WTFMove(promise));
}

void FetchBody::consumeOnceLoadingFinished(FetchBodyConsumer::Type type, Ref<DeferredPromise>&& promise)
{
    m_consumer.setType(type);
    m_consumer.setConsumePromise(WTFMove(promise));
}

void FetchBody::consume(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise)
{
    if (isArrayBuffer()) {
        consumeArrayBuffer(owner, WTFMove(promise));
        return;
    }
    if (isArrayBufferView()) {
        consumeArrayBufferView(owner, WTFMove(promise));
        return;
    }
    if (isText()) {
        consumeText(owner, WTFMove(promise), textBody());
        return;
    }
    if (isURLSearchParams()) {
        consumeText(owner, WTFMove(promise), urlSearchParamsBody().toString());
        return;
    }
    if (isBlob()) {
        consumeBlob(owner, WTFMove(promise));
        return;
    }
    if (isFormData()) {
        consumeFormData(owner, WTFMove(promise));
        return;
    }

    m_consumer.resolve(WTFMove(promise), owner.contentType(), &owner, m_readableStream.get());
}

void FetchBody::consumeAsStream(FetchBodyOwner& owner, FetchBodySource& source)
{
    bool closeStream = false;
    if (isArrayBuffer())
        closeStream = source.enqueue(ArrayBuffer::tryCreate(arrayBufferBody().span()));
    else if (isArrayBufferView())
        closeStream = source.enqueue(ArrayBuffer::tryCreate(arrayBufferViewBody().span()));
    else if (isText()) {
        auto data = PAL::TextCodecUTF8::encodeUTF8(textBody());
        closeStream = source.enqueue(ArrayBuffer::tryCreate(data));
    } else if (isURLSearchParams()) {
        auto data = PAL::TextCodecUTF8::encodeUTF8(urlSearchParamsBody().toString());
        closeStream = source.enqueue(ArrayBuffer::tryCreate(data));
    } else if (isBlob())
        owner.loadBlob(blobBody(), nullptr);
    else if (isFormData())
        m_consumer.consumeFormDataAsStream(formDataBody(), source, owner.scriptExecutionContext());
    else if (m_consumer.hasData())
        closeStream = source.enqueue(m_consumer.takeAsArrayBuffer());
    else
        closeStream = true;

    if (closeStream)
        source.close();
}

void FetchBody::consumeArrayBuffer(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise)
{
    m_consumer.resolveWithData(WTFMove(promise), owner.contentType(), arrayBufferBody().span());
    m_data = nullptr;
}

void FetchBody::consumeArrayBufferView(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise)
{
    m_consumer.resolveWithData(WTFMove(promise), owner.contentType(), arrayBufferViewBody().span());
    m_data = nullptr;
}

void FetchBody::consumeText(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise, const String& text)
{
    auto data = PAL::TextCodecUTF8::encodeUTF8(text);
    m_consumer.resolveWithData(WTFMove(promise), owner.contentType(), data.span());
    m_data = nullptr;
}

void FetchBody::consumeBlob(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise)
{
    m_consumer.setConsumePromise(WTFMove(promise));
    owner.loadBlob(blobBody(), &m_consumer);
    m_data = nullptr;
}

void FetchBody::consumeFormData(FetchBodyOwner& owner, Ref<DeferredPromise>&& promise)
{
    m_consumer.resolveWithFormData(WTFMove(promise), owner.contentType(), formDataBody(), owner.scriptExecutionContext());
    m_data = nullptr;
}

void FetchBody::loadingFailed(const Exception& exception)
{
    m_consumer.loadingFailed(exception);
}

void FetchBody::loadingSucceeded(const String& contentType)
{
    m_consumer.loadingSucceeded(contentType);
}

RefPtr<FormData> FetchBody::bodyAsFormData() const
{
    if (isText())
        return FormData::create(PAL::TextCodecUTF8::encodeUTF8(textBody()));
    if (isURLSearchParams())
        return FormData::create(PAL::TextCodecUTF8::encodeUTF8(urlSearchParamsBody().toString()));
    if (isBlob()) {
        auto body = FormData::create();
        body->appendBlob(blobBody().url());
        return body;
    }
    if (isArrayBuffer())
        return FormData::create(arrayBufferBody().span());
    if (isArrayBufferView())
        return FormData::create(arrayBufferViewBody().span());
    if (isFormData())
        return &const_cast<FormData&>(formDataBody());
    if (auto* data = m_consumer.data())
        return FormData::create(data->makeContiguous()->span());

    ASSERT_NOT_REACHED();
    return nullptr;
}

void FetchBody::convertReadableStreamToArrayBuffer(FetchBodyOwner& owner, CompletionHandler<void(std::optional<Exception>&&)>&& completionHandler)
{
    ASSERT(hasReadableStream());

    consumer().extract(*readableStream(), [owner = Ref { owner }, data = SharedBufferBuilder(), completionHandler = WTFMove(completionHandler)](auto&& result) mutable {
        if (result.hasException()) {
            completionHandler(result.releaseException());
            return;
        }

        if (auto* chunk = result.returnValue()) {
            data.append(*chunk);
            return;
        }

        if (RefPtr arrayBuffer = data.takeAsArrayBuffer())
            owner->body().m_data = *arrayBuffer;

        completionHandler({ });
    });
}

FetchBody::TakenData FetchBody::take()
{
    if (m_consumer.hasData()) {
        auto buffer = m_consumer.takeData();
        if (!buffer)
            return nullptr;
        return buffer->makeContiguous();
    }

    if (isBlob()) {
        auto body = FormData::create();
        body->appendBlob(blobBody().url());
        return TakenData { WTFMove(body) };
    }

    if (isFormData())
        return formDataBody();

    if (isText())
        return SharedBuffer::create(PAL::TextCodecUTF8::encodeUTF8(textBody()));
    if (isURLSearchParams())
        return SharedBuffer::create(PAL::TextCodecUTF8::encodeUTF8(urlSearchParamsBody().toString()));

    if (isArrayBuffer())
        return SharedBuffer::create(arrayBufferBody().span());
    if (isArrayBufferView())
        return SharedBuffer::create(arrayBufferViewBody().span());

    return nullptr;
}

FetchBody FetchBody::clone()
{
    FetchBody clone(m_consumer.clone());

    if (isArrayBuffer())
        clone.m_data = arrayBufferBody();
    else if (isArrayBufferView())
        clone.m_data = arrayBufferViewBody();
    else if (isBlob())
        clone.m_data = blobBody();
    else if (isFormData())
        clone.m_data = const_cast<FormData&>(formDataBody());
    else if (isText())
        clone.m_data = textBody();
    else if (isURLSearchParams())
        clone.m_data = urlSearchParamsBody();
    else if (m_readableStream) {
        auto clones = m_readableStream->tee(true);
        ASSERT(!clones.hasException());
        if (!clones.hasException()) {
            auto pair = clones.releaseReturnValue();
            m_readableStream = WTFMove(pair[0]);
            clone.m_readableStream = WTFMove(pair[1]);
        }
    }
    return clone;
}

}
