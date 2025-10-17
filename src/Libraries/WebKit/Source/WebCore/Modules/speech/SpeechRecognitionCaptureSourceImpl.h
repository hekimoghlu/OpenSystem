/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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

#if ENABLE(MEDIA_STREAM)

#include "RealtimeMediaSource.h"
#include "SpeechRecognitionConnectionClientIdentifier.h"
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>

#if PLATFORM(COCOA)
#include "AudioSampleDataSource.h"
#endif

namespace WebCore {
class SpeechRecognitionCaptureSourceImpl;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::SpeechRecognitionCaptureSourceImpl> : std::true_type { };
}

namespace WTF {
class MediaTime;
}

namespace WebCore {

class AudioStreamDescription;
class PlatformAudioData;
class SpeechRecognitionUpdate;
enum class SpeechRecognitionUpdateType : uint8_t;

class SpeechRecognitionCaptureSourceImpl final
    : public RealtimeMediaSourceObserver
    , public RealtimeMediaSource::AudioSampleObserver
    , public CanMakeCheckedPtr<SpeechRecognitionCaptureSourceImpl> {
    WTF_MAKE_TZONE_ALLOCATED(SpeechRecognitionCaptureSourceImpl);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SpeechRecognitionCaptureSourceImpl);
public:
    using DataCallback = Function<void(const WTF::MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t)>;
    using StateUpdateCallback = Function<void(const SpeechRecognitionUpdate&)>;
    SpeechRecognitionCaptureSourceImpl(SpeechRecognitionConnectionClientIdentifier, DataCallback&&, StateUpdateCallback&&, Ref<RealtimeMediaSource>&&);
    ~SpeechRecognitionCaptureSourceImpl();
    void mute();

private:
    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    // RealtimeMediaSource::AudioSampleObserver
    void audioSamplesAvailable(const WTF::MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t) final;

#if PLATFORM(COCOA)
    void pullSamplesAndCallDataCallback(AudioSampleDataSource*, const WTF::MediaTime&, const CAAudioStreamDescription&, size_t sampleCount);
#endif

    // RealtimeMediaSourceObserver
    void sourceStarted() final;
    void sourceStopped() final;
    void sourceMutedChanged() final;

    SpeechRecognitionConnectionClientIdentifier m_clientIdentifier;
    DataCallback m_dataCallback;
    StateUpdateCallback m_stateUpdateCallback;
    Ref<RealtimeMediaSource> m_source;

#if PLATFORM(COCOA)
    RefPtr<AudioSampleDataSource> m_dataSource WTF_GUARDED_BY_LOCK(m_dataSourceLock);
    Lock m_dataSourceLock;
#endif
};

} // namespace WebCore

#endif

