/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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

#include "AudioEncoder.h"
#include "JSDOMPromiseDeferredForward.h"
#include "WebCodecsAudioEncoderConfig.h"
#include "WebCodecsBase.h"
#include <wtf/Vector.h>

namespace WebCore {

class WebCodecsEncodedAudioChunk;
class WebCodecsErrorCallback;
class WebCodecsEncodedAudioChunkOutputCallback;
class WebCodecsAudioData;
struct WebCodecsEncodedAudioChunkMetadata;

class WebCodecsAudioEncoder : public WebCodecsBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebCodecsAudioEncoder);
public:
    ~WebCodecsAudioEncoder();

    struct Init {
        RefPtr<WebCodecsEncodedAudioChunkOutputCallback> output;
        RefPtr<WebCodecsErrorCallback> error;
    };

    static Ref<WebCodecsAudioEncoder> create(ScriptExecutionContext&, Init&&);

    size_t encodeQueueSize() const { return codecQueueSize(); }

    ExceptionOr<void> configure(ScriptExecutionContext&, WebCodecsAudioEncoderConfig&&);
    ExceptionOr<void> encode(Ref<WebCodecsAudioData>&&);
    void flush(Ref<DeferredPromise>&&);
    ExceptionOr<void> reset();
    ExceptionOr<void> close();

    static void isConfigSupported(ScriptExecutionContext&, WebCodecsAudioEncoderConfig&&, Ref<DeferredPromise>&&);

    WebCodecsEncodedAudioChunkOutputCallback& outputCallbackConcurrently() { return m_output.get(); }
    WebCodecsErrorCallback& errorCallbackConcurrently() { return m_error.get(); }
private:
    WebCodecsAudioEncoder(ScriptExecutionContext&, Init&&);

    // ActiveDOMObject.
    void stop() final;
    void suspend(ReasonForSuspension) final;

    // EventTarget.
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::WebCodecsAudioEncoder; }

    ExceptionOr<void> closeEncoder(Exception&&);
    ExceptionOr<void> resetEncoder(const Exception&);
    void setInternalEncoder(Ref<AudioEncoder>&&);

    WebCodecsEncodedAudioChunkMetadata createEncodedChunkMetadata();

    size_t m_encodeQueueSize { 0 };
    Ref<WebCodecsEncodedAudioChunkOutputCallback> m_output;
    Ref<WebCodecsErrorCallback> m_error;
    RefPtr<AudioEncoder> m_internalEncoder;
    Vector<Ref<DeferredPromise>> m_pendingFlushPromises;
    bool m_isKeyChunkRequired { false };
    WebCodecsAudioEncoderConfig m_baseConfiguration;
    AudioEncoder::ActiveConfiguration m_activeConfiguration;
    bool m_hasNewActiveConfiguration { false };
    size_t m_encoderCount { 0 };
};

}

#endif
