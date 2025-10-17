/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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

#if USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/MonotonicTime.h>
#include <wtf/WorkQueue.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/modules/audio_device/include/audio_device.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {
class BaseAudioMediaStreamTrackRendererUnit;
class IncomingAudioMediaStreamTrackRendererUnit;

// LibWebRTCAudioModule is pulling streamed data to ensure audio data is passed to the audio track.
class LibWebRTCAudioModule : public webrtc::AudioDeviceModule, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<LibWebRTCAudioModule, WTF::DestructionThread::Main> {
public:
    static Ref<LibWebRTCAudioModule> create() { return adoptRef(*new LibWebRTCAudioModule()); }
    ~LibWebRTCAudioModule();

    static constexpr unsigned PollSamplesCount = 1;

#if PLATFORM(COCOA)
    void startIncomingAudioRendering() { ++m_isRenderingIncomingAudioCounter; }
    void stopIncomingAudioRendering() { --m_isRenderingIncomingAudioCounter; }

    BaseAudioMediaStreamTrackRendererUnit& incomingAudioMediaStreamTrackRendererUnit();
    uint64_t currentAudioSampleCount() const { return m_currentAudioSampleCount; }
#endif

    void AddRef() const final { ref(); }
    webrtc::RefCountReleaseStatus Release() const final
    {
        deref();
        return webrtc::RefCountReleaseStatus::kOtherRefsRemained;
    }

private:
    LibWebRTCAudioModule();

    template<typename U> U shouldNotBeCalled(U value) const
    {
        ASSERT_NOT_REACHED();
        return value;
    }

    // webrtc::AudioDeviceModule API
    int32_t StartPlayout() final;
    int32_t StopPlayout() final;
    int32_t RegisterAudioCallback(webrtc::AudioTransport*) final;
    bool Playing() const final { return m_isPlaying; }

    int32_t ActiveAudioLayer(AudioLayer*) const final { return shouldNotBeCalled(-1); }
    int32_t Init() final { return 0; }
    int32_t Terminate() final { return 0; }
    bool Initialized() const final { return true; }
    int16_t PlayoutDevices() final { return 0; }
    int16_t RecordingDevices() final { return 0; }
    int32_t PlayoutDeviceName(uint16_t, char[webrtc::kAdmMaxDeviceNameSize], char[webrtc::kAdmMaxGuidSize]) final { return 0; }
    int32_t RecordingDeviceName(uint16_t, char[webrtc::kAdmMaxDeviceNameSize], char[webrtc::kAdmMaxGuidSize]) final { return 0; }
    int32_t SetPlayoutDevice(uint16_t) final { return 0; }
    int32_t SetPlayoutDevice(WindowsDeviceType) final { return 0; }
    int32_t SetRecordingDevice(uint16_t) final { return 0; }
    int32_t SetRecordingDevice(WindowsDeviceType) final { return 0; }
    int32_t PlayoutIsAvailable(bool*) final { return shouldNotBeCalled(-1); }
    int32_t InitPlayout() final { return 0; }
    bool PlayoutIsInitialized() const final { return true; }
    int32_t RecordingIsAvailable(bool*) final { return shouldNotBeCalled(-1); }
    int32_t InitRecording() final { return 0; }
    bool RecordingIsInitialized() const final { return false; }
    int32_t StartRecording() final { return 0; }
    int32_t StopRecording() final { return 0;  }
    bool Recording() const final { return 0;  }
    int32_t InitSpeaker() final { return 0; }
    bool SpeakerIsInitialized() const final { return false; }
    int32_t InitMicrophone() final { return 0; }
    bool MicrophoneIsInitialized() const final { return false; }
    int32_t MicrophoneVolumeIsAvailable(bool*) final { return shouldNotBeCalled(-1); }
    int32_t SpeakerVolumeIsAvailable(bool*) final { return shouldNotBeCalled(-1); }
    int32_t SetSpeakerVolume(uint32_t) final { return shouldNotBeCalled(-1); }
    int32_t SpeakerVolume(uint32_t*) const final { return shouldNotBeCalled(-1); }
    int32_t MaxSpeakerVolume(uint32_t*) const final { return shouldNotBeCalled(-1); }
    int32_t MinSpeakerVolume(uint32_t*) const final { return shouldNotBeCalled(-1); }
    int32_t SetMicrophoneVolume(uint32_t) final { return shouldNotBeCalled(-1); }
    int32_t MicrophoneVolume(uint32_t*) const final { return shouldNotBeCalled(-1); }
    int32_t MaxMicrophoneVolume(uint32_t*) const final { return shouldNotBeCalled(-1); }
    int32_t MinMicrophoneVolume(uint32_t*) const final { return shouldNotBeCalled(-1); }
    int32_t SpeakerMuteIsAvailable(bool*) final { return shouldNotBeCalled(-1); }
    int32_t SetSpeakerMute(bool) final { return shouldNotBeCalled(-1); }
    int32_t SpeakerMute(bool*) const final { return shouldNotBeCalled(-1); }
    int32_t MicrophoneMuteIsAvailable(bool*) final { return shouldNotBeCalled(-1); }
    int32_t SetMicrophoneMute(bool) final { return shouldNotBeCalled(-1); }
    int32_t MicrophoneMute(bool*) const final { return shouldNotBeCalled(-1); }
    int32_t StereoPlayoutIsAvailable(bool* available) const final { *available = false; return 0; }
    int32_t SetStereoPlayout(bool) final { return 0; }
    int32_t StereoPlayout(bool*) const final { return shouldNotBeCalled(-1); }
    int32_t StereoRecordingIsAvailable(bool* available) const final { *available = false; return 0;  }
    int32_t SetStereoRecording(bool) final { return 0;  }
    int32_t StereoRecording(bool*) const final { return shouldNotBeCalled(-1); }
    int32_t PlayoutDelay(uint16_t* delay) const final { *delay = 0; return 0; }
    bool BuiltInAECIsAvailable() const final { return false; }
    bool BuiltInAGCIsAvailable() const final { return false;  }
    bool BuiltInNSIsAvailable() const final { return false;  }
    int32_t EnableBuiltInAEC(bool) final { return shouldNotBeCalled(-1); }
    int32_t EnableBuiltInAGC(bool) final { return shouldNotBeCalled(-1); }
    int32_t EnableBuiltInNS(bool) final { return shouldNotBeCalled(-1); }

#if defined(WEBRTC_IOS)
    int GetPlayoutAudioParameters(webrtc::AudioParameters*) const final { return shouldNotBeCalled(-1); }
    int GetRecordAudioParameters(webrtc::AudioParameters*) const final { return shouldNotBeCalled(-1); }
#endif

private:
    void pollAudioData();
    void pollFromSource();
    void logTimerFired();
    Seconds computeDelayUntilNextPolling();

    static constexpr Seconds logTimerInterval = 2_s;

    const Ref<WorkQueue> m_queue;
    bool m_isPlaying { false };
    webrtc::AudioTransport* m_audioTransport { nullptr };
    MonotonicTime m_pollingTime;
    Timer m_logTimer;
    int m_timeSpent { 0 };

#if PLATFORM(COCOA)
    uint64_t m_currentAudioSampleCount { 0 };
    std::atomic<uint64_t> m_isRenderingIncomingAudioCounter { 0 };
    std::unique_ptr<IncomingAudioMediaStreamTrackRendererUnit> m_incomingAudioMediaStreamTrackRendererUnit;
#endif
};

} // namespace WebCore

#endif // USE(LIBWEBRTC)
