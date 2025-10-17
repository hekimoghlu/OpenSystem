/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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

#include "JSDOMPromiseDeferredForward.h"
#include "VideoDecoder.h"
#include "WebCodecsBase.h"
#include "WebCodecsEncodedVideoChunkType.h"
#include "WebCodecsVideoDecoderSupport.h"
#include <wtf/Vector.h>

namespace WebCore {

class WebCodecsEncodedVideoChunk;
class WebCodecsErrorCallback;
class WebCodecsVideoFrameOutputCallback;

class WebCodecsVideoDecoder : public WebCodecsBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebCodecsVideoDecoder);
public:
    ~WebCodecsVideoDecoder();

    struct Init {
        RefPtr<WebCodecsVideoFrameOutputCallback> output;
        RefPtr<WebCodecsErrorCallback> error;
    };

    static Ref<WebCodecsVideoDecoder> create(ScriptExecutionContext&, Init&&);

    size_t decodeQueueSize() const { return codecQueueSize(); }

    WebCodecsVideoFrameOutputCallback& outputCallbackConcurrently() { return m_output.get(); }
    WebCodecsErrorCallback& errorCallbackConcurrently() { return m_error.get(); }

    ExceptionOr<void> configure(ScriptExecutionContext&, WebCodecsVideoDecoderConfig&&);
    ExceptionOr<void> decode(Ref<WebCodecsEncodedVideoChunk>&&);
    ExceptionOr<void> flush(Ref<DeferredPromise>&&);
    ExceptionOr<void> reset();
    ExceptionOr<void> close();

    static void isConfigSupported(ScriptExecutionContext&, WebCodecsVideoDecoderConfig&&, Ref<DeferredPromise>&&);

private:
    WebCodecsVideoDecoder(ScriptExecutionContext&, Init&&);
    size_t maximumCodecOperationsEnqueued() const final { return 4; }

    // ActiveDOMObject.
    void stop() final;
    void suspend(ReasonForSuspension) final;

    // EventTarget.
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::WebCodecsVideoDecoder; }

    ExceptionOr<void> closeDecoder(Exception&&);
    ExceptionOr<void> resetDecoder(const Exception&);
    void setInternalDecoder(Ref<VideoDecoder>&&);

    Ref<WebCodecsVideoFrameOutputCallback> m_output;
    Ref<WebCodecsErrorCallback> m_error;
    RefPtr<VideoDecoder> m_internalDecoder;
    Vector<Ref<DeferredPromise>> m_pendingFlushPromises;
    bool m_isKeyChunkRequired { false };
    size_t m_decoderCount { 0 };
};

}

#endif
