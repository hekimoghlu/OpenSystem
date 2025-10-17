/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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

#include "FetchBodySource.h"
#include "FormDataConsumer.h"
#include "JSDOMPromiseDeferredForward.h"
#include "ReadableStreamSink.h"
#include "ScriptExecutionContextIdentifier.h"
#include "SharedBuffer.h"
#include "UserGestureIndicator.h"

namespace WebCore {

class Blob;
class DOMFormData;
class FetchBodyOwner;
class FetchBodySource;
class FormData;
class ReadableStream;

class FetchBodyConsumer final : public CanMakeCheckedPtr<FetchBodyConsumer> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(FetchBodyConsumer);
public:
    enum class Type { None, ArrayBuffer, Blob, Bytes, JSON, Text, FormData };

    explicit FetchBodyConsumer(Type);
    FetchBodyConsumer(FetchBodyConsumer&&);
    ~FetchBodyConsumer();
    FetchBodyConsumer& operator=(FetchBodyConsumer&&);

    FetchBodyConsumer clone();

    void append(const SharedBuffer&);

    bool hasData() const { return !!m_buffer; }
    const FragmentedSharedBuffer* data() const { return m_buffer.get().get(); }
    void setData(Ref<FragmentedSharedBuffer>&&);

    RefPtr<FragmentedSharedBuffer> takeData();
    RefPtr<JSC::ArrayBuffer> takeAsArrayBuffer();
    String takeAsText();

    bool hasPendingActivity() const { return m_formDataConsumer ? m_formDataConsumer->hasPendingActivity() : false; }

    void setType(Type type) { m_type = type; }

    void clean();

    void extract(ReadableStream&, ReadableStreamToSharedBufferSink::Callback&&);
    void resolve(Ref<DeferredPromise>&&, const String& contentType, FetchBodyOwner*, ReadableStream*);
    void resolveWithData(Ref<DeferredPromise>&&, const String& contentType, std::span<const uint8_t>);
    void resolveWithFormData(Ref<DeferredPromise>&&, const String& contentType, const FormData&, ScriptExecutionContext*);
    void consumeFormDataAsStream(const FormData&, FetchBodySource&, ScriptExecutionContext*);

    void loadingFailed(const Exception&);
    void loadingSucceeded(const String& contentType);

    void setConsumePromise(Ref<DeferredPromise>&&);
    void setSource(Ref<FetchBodySource>&&);

    void setAsLoading() { m_isLoading = true; }

    static RefPtr<DOMFormData> packageFormData(ScriptExecutionContext*, const String& contentType, std::span<const uint8_t> data);

private:
    Ref<Blob> takeAsBlob(ScriptExecutionContext*, const String& contentType);
    void resetConsumePromise();

    Type m_type;
    SharedBufferBuilder m_buffer;
    RefPtr<DeferredPromise> m_consumePromise;
    RefPtr<ReadableStreamToSharedBufferSink> m_sink;
    RefPtr<FetchBodySource> m_source;
    bool m_isLoading { false };
    RefPtr<UserGestureToken> m_userGestureToken;
    RefPtr<FormDataConsumer> m_formDataConsumer;
};

} // namespace WebCore
