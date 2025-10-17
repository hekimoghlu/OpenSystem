/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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

#include "SpeechRecognitionCaptureSource.h"
#include "SpeechRecognitionConnectionClientIdentifier.h"
#include "SpeechRecognitionError.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

#if HAVE(SPEECHRECOGNIZER)
#include <CoreMedia/CMTime.h>
#include <wtf/RetainPtr.h>
OBJC_CLASS WebSpeechRecognizerTask;
#endif

namespace WebCore {
class SpeechRecognizer;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::SpeechRecognizer> : std::true_type { };
}

namespace WebCore {

class SpeechRecognitionRequest;
class SpeechRecognitionUpdate;

class SpeechRecognizer : public CanMakeWeakPtr<SpeechRecognizer> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(SpeechRecognizer, WEBCORE_EXPORT);
public:
    using DelegateCallback = Function<void(const SpeechRecognitionUpdate&)>;
    WEBCORE_EXPORT explicit SpeechRecognizer(DelegateCallback&&, UniqueRef<SpeechRecognitionRequest>&&);

#if ENABLE(MEDIA_STREAM)
    WEBCORE_EXPORT void start(Ref<RealtimeMediaSource>&&, bool mockSpeechRecognitionEnabled);
#endif
    WEBCORE_EXPORT void abort(std::optional<SpeechRecognitionError>&& = std::nullopt);
    WEBCORE_EXPORT void stop();
    WEBCORE_EXPORT void prepareForDestruction();

    WEBCORE_EXPORT SpeechRecognitionConnectionClientIdentifier clientIdentifier() const;
    SpeechRecognitionCaptureSource* source() { return m_source.get(); }

    void setInactive() { m_state = State::Inactive; }

private:
    enum class State {
        Inactive,
        Running,
        Stopping,
        Aborting
    };

#if ENABLE(MEDIA_STREAM)
    void startCapture(Ref<RealtimeMediaSource>&&);
#endif
    void stopCapture();
    void dataCaptured(const WTF::MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t sampleCount);
    bool startRecognition(bool mockSpeechRecognitionEnabled, SpeechRecognitionConnectionClientIdentifier, const String& localeIdentifier, bool continuous, bool interimResults, uint64_t alternatives);
    void abortRecognition();
    void stopRecognition();

    DelegateCallback m_delegateCallback;
    UniqueRef<SpeechRecognitionRequest> m_request;
    std::unique_ptr<SpeechRecognitionCaptureSource> m_source;
    State m_state { State::Inactive };

#if HAVE(SPEECHRECOGNIZER)
    RetainPtr<WebSpeechRecognizerTask> m_task;
    CMTime m_currentAudioSampleTime;
#endif
};

} // namespace WebCore
