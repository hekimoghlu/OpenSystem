/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
#include "VideoEncoder.h"
#include "WebCodecsBase.h"
#include "WebCodecsVideoEncoderConfig.h"
#include <wtf/Vector.h>

namespace WebCore {

class WebCodecsEncodedVideoChunk;
class WebCodecsErrorCallback;
class WebCodecsEncodedVideoChunkOutputCallback;
class WebCodecsVideoFrame;
struct WebCodecsEncodedVideoChunkMetadata;
struct WebCodecsVideoEncoderEncodeOptions;

class WebCodecsVideoEncoder : public WebCodecsBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebCodecsVideoEncoder);
public:
    ~WebCodecsVideoEncoder();

    struct Init {
        RefPtr<WebCodecsEncodedVideoChunkOutputCallback> output;
        RefPtr<WebCodecsErrorCallback> error;
    };

    static Ref<WebCodecsVideoEncoder> create(ScriptExecutionContext&, Init&&);

    size_t encodeQueueSize() const { return codecQueueSize(); }

    ExceptionOr<void> configure(ScriptExecutionContext&, WebCodecsVideoEncoderConfig&&);
    ExceptionOr<void> encode(Ref<WebCodecsVideoFrame>&&, WebCodecsVideoEncoderEncodeOptions&&);
    void flush(Ref<DeferredPromise>&&);
    ExceptionOr<void> reset();
    ExceptionOr<void> close();

    static void isConfigSupported(ScriptExecutionContext&, WebCodecsVideoEncoderConfig&&, Ref<DeferredPromise>&&);

    WebCodecsEncodedVideoChunkOutputCallback& outputCallbackConcurrently() { return m_output.get(); }
    WebCodecsErrorCallback& errorCallbackConcurrently() { return m_error.get(); }

private:
    WebCodecsVideoEncoder(ScriptExecutionContext&, Init&&);
    size_t maximumCodecOperationsEnqueued() const final { return 4; }

    // ActiveDOMObject.
    void stop() final;
    void suspend(ReasonForSuspension) final;

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::WebCodecsVideoEncoder; }

    ExceptionOr<void> closeEncoder(Exception&&);
    ExceptionOr<void> resetEncoder(const Exception&);
    void setInternalEncoder(Ref<VideoEncoder>&&);

    WebCodecsEncodedVideoChunkMetadata createEncodedChunkMetadata(std::optional<unsigned>);
    void updateRates(const WebCodecsVideoEncoderConfig&);

    Ref<WebCodecsEncodedVideoChunkOutputCallback> m_output;
    Ref<WebCodecsErrorCallback> m_error;
    RefPtr<VideoEncoder> m_internalEncoder;
    Vector<Ref<DeferredPromise>> m_pendingFlushPromises;
    bool m_isKeyChunkRequired { false };
    WebCodecsVideoEncoderConfig m_baseConfiguration;
    VideoEncoder::ActiveConfiguration m_activeConfiguration;
    bool m_hasNewActiveConfiguration { false };
    size_t m_encoderCount { 0 };
};

}

#endif
