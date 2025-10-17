/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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

#if ENABLE(WEB_CODECS)

#include "AudioDecoder.h"
#include "JSDOMPromiseDeferredForward.h"
#include "WebCodecsAudioDecoderConfig.h"
#include "WebCodecsAudioDecoderSupport.h"
#include "WebCodecsBase.h"
#include "WebCodecsEncodedAudioChunkType.h"
#include <wtf/Vector.h>

namespace WebCore {

class WebCodecsEncodedAudioChunk;
class WebCodecsErrorCallback;
class WebCodecsAudioDataOutputCallback;

class WebCodecsAudioDecoder : public WebCodecsBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebCodecsAudioDecoder);
public:
    ~WebCodecsAudioDecoder();

    struct Init {
        RefPtr<WebCodecsAudioDataOutputCallback> output;
        RefPtr<WebCodecsErrorCallback> error;
    };

    static Ref<WebCodecsAudioDecoder> create(ScriptExecutionContext&, Init&&);

    size_t decodeQueueSize() const { return codecQueueSize(); }

    ExceptionOr<void> configure(ScriptExecutionContext&, WebCodecsAudioDecoderConfig&&);
    ExceptionOr<void> decode(Ref<WebCodecsEncodedAudioChunk>&&);
    ExceptionOr<void> flush(Ref<DeferredPromise>&&);
    ExceptionOr<void> reset();
    ExceptionOr<void> close();

    static void isConfigSupported(ScriptExecutionContext&, WebCodecsAudioDecoderConfig&&, Ref<DeferredPromise>&&);

    WebCodecsAudioDataOutputCallback& outputCallbackConcurrently() { return m_output.get(); }
    WebCodecsErrorCallback& errorCallbackConcurrently() { return m_error.get(); }

private:
    WebCodecsAudioDecoder(ScriptExecutionContext&, Init&&);

    // ActiveDOMObject.
    void stop() final;
    void suspend(ReasonForSuspension) final;

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::WebCodecsAudioDecoder; }

    ExceptionOr<void> closeDecoder(Exception&&);
    ExceptionOr<void> resetDecoder(const Exception&);
    void setInternalDecoder(Ref<AudioDecoder>&&);

    Ref<WebCodecsAudioDataOutputCallback> m_output;
    Ref<WebCodecsErrorCallback> m_error;
    RefPtr<AudioDecoder> m_internalDecoder;
    Vector<Ref<DeferredPromise>> m_pendingFlushPromises;
    bool m_isKeyChunkRequired { false };
    size_t m_decoderCount { 0 };
};

}

#endif
