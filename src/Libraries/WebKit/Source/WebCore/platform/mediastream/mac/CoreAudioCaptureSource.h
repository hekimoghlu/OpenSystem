/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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

#include "AudioSession.h"
#include "CAAudioStreamDescription.h"
#include "CaptureDevice.h"
#include "RealtimeMediaSource.h"
#include "RealtimeMediaSourceFactory.h"
#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudioTypes.h>
#include <wtf/CheckedRef.h>
#include <wtf/text/WTFString.h>

typedef struct OpaqueCMClock* CMClockRef;

namespace WTF {
class MediaTime;
}

namespace WebCore {

class AudioSampleBufferList;
class AudioSampleDataSource;
class BaseAudioSharedUnit;
class CaptureDeviceInfo;
class WebAudioSourceProviderAVFObjC;

class CoreAudioCaptureSource : public RealtimeMediaSource, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<CoreAudioCaptureSource, WTF::DestructionThread::MainRunLoop> {
public:
    WEBCORE_EXPORT static CaptureSourceOrError create(const CaptureDevice&, MediaDeviceHashSalts&&, const MediaConstraints*, std::optional<PageIdentifier>);
    static CaptureSourceOrError createForTesting(String&& deviceID, AtomString&& label, MediaDeviceHashSalts&&, const MediaConstraints*, std::optional<PageIdentifier>, std::optional<bool>);

    WEBCORE_EXPORT static AudioCaptureFactory& factory();

    CMClockRef timebaseClock();

    void handleNewCurrentMicrophoneDevice(const CaptureDevice&);

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;
    virtual ~CoreAudioCaptureSource();

protected:
    CoreAudioCaptureSource(const CaptureDevice&, uint32_t, MediaDeviceHashSalts&&, std::optional<PageIdentifier>);

    bool canResumeAfterInterruption() const { return m_canResumeAfterInterruption; }
    void setCanResumeAfterInterruption(bool value) { m_canResumeAfterInterruption = value; }

private:
    friend class BaseAudioSharedUnit;
    friend class CoreAudioSharedUnit;
    friend class CoreAudioCaptureSourceFactory;

    bool isCaptureSource() const final { return true; }
    void startProducingData() final;
    void stopProducingData() final;
    void endProducingData() final;

    void delaySamples(Seconds) final;
#if PLATFORM(IOS_FAMILY)
    void setIsInBackground(bool) final;
#endif

    std::optional<Vector<int>> discreteSampleRates() const final { return { { 8000, 16000, 32000, 44100, 48000, 96000 } }; }

    const RealtimeMediaSourceCapabilities& capabilities() final;
    const RealtimeMediaSourceSettings& settings() final;
    void settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag>) final;

    bool interrupted() const final;
    CaptureDevice::DeviceType deviceType() const final { return CaptureDevice::DeviceType::Microphone; }

    void initializeToStartProducingData();
    void audioUnitWillStart();

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const override { return "CoreAudioCaptureSource"_s; }
#endif

    uint32_t m_captureDeviceID { 0 };

    std::optional<RealtimeMediaSourceCapabilities> m_capabilities;
    std::optional<RealtimeMediaSourceSettings> m_currentSettings;

    bool m_canResumeAfterInterruption { true };
    bool m_isReadyToStart { false };

    std::optional<bool> m_echoCancellationCapability;
    BaseAudioSharedUnit* m_overrideUnit { nullptr };
};

class CoreAudioSpeakerSamplesProducer {
public:
    virtual ~CoreAudioSpeakerSamplesProducer() = default;
    // Main thread
    virtual const CAAudioStreamDescription& format() = 0;
    virtual void captureUnitIsStarting() = 0;
    virtual void captureUnitHasStopped() = 0;
    virtual void canRenderAudioChanged() = 0;
    // Background thread.
    virtual OSStatus produceSpeakerSamples(size_t sampleCount, AudioBufferList&, uint64_t sampleTime, double hostTime, AudioUnitRenderActionFlags&) = 0;
};

class CoreAudioCaptureSourceFactory : public AudioCaptureFactory, public AudioSessionInterruptionObserver {
public:
    WEBCORE_EXPORT static CoreAudioCaptureSourceFactory& singleton();

    CoreAudioCaptureSourceFactory();
    ~CoreAudioCaptureSourceFactory();

    void scheduleReconfiguration();

    WEBCORE_EXPORT void registerSpeakerSamplesProducer(CoreAudioSpeakerSamplesProducer&);
    WEBCORE_EXPORT void unregisterSpeakerSamplesProducer(CoreAudioSpeakerSamplesProducer&);
    WEBCORE_EXPORT bool isAudioCaptureUnitRunning();
    WEBCORE_EXPORT bool shouldAudioCaptureUnitRenderAudio();

private:
    // AudioSessionInterruptionObserver
    void beginAudioSessionInterruption() final { beginInterruption(); }
    void endAudioSessionInterruption(AudioSession::MayResume) final { endInterruption(); }

    // AudioCaptureFactory
    CaptureSourceOrError createAudioCaptureSource(const CaptureDevice&, MediaDeviceHashSalts&&, const MediaConstraints*, std::optional<PageIdentifier>) override;
    CaptureDeviceManager& audioCaptureDeviceManager() override;
    const Vector<CaptureDevice>& speakerDevices() const override;
    void enableMutedSpeechActivityEventListener(Function<void()>&&) final;
    void disableMutedSpeechActivityEventListener() final;

    void beginInterruption();
    void endInterruption();
};

inline CaptureSourceOrError CoreAudioCaptureSourceFactory::createAudioCaptureSource(const CaptureDevice& device, MediaDeviceHashSalts&& hashSalts, const MediaConstraints* constraints, std::optional<PageIdentifier> pageIdentifier)
{
    return CoreAudioCaptureSource::create(device, WTFMove(hashSalts), constraints, pageIdentifier);
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
