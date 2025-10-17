/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 9, 2022.
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

#include "RealtimeMediaSourceCapabilities.h"
#include "RealtimeMediaSourceCenter.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Function.h>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/MediaTime.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class BaseAudioSharedUnit;
}

namespace WebCore {

class AudioStreamDescription;
class CaptureDevice;
class CoreAudioCaptureSource;
class PlatformAudioData;

class BaseAudioSharedUnit : public RefCounted<BaseAudioSharedUnit>, public RealtimeMediaSourceCenterObserver {
public:
    virtual ~BaseAudioSharedUnit();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void startProducingData();
    void stopProducingData();
    WEBCORE_EXPORT void reconfigure();
    virtual bool isProducingData() const = 0;

    virtual void delaySamples(Seconds) { }
    virtual void prewarmAudioUnitCreation(CompletionHandler<void()>&& callback) { callback(); };

    void prepareForNewCapture();

    OSStatus resume();
    OSStatus suspend();

    bool isSuspended() const { return m_suspended; }

    double volume() const { return m_volume; }
    int sampleRate() const { return m_sampleRate; }
    bool enableEchoCancellation() const { return m_enableEchoCancellation; }

    void setVolume(double volume) { m_volume = volume; }
    void setSampleRate(int sampleRate) { m_sampleRate = sampleRate; }
    void setEnableEchoCancellation(bool enableEchoCancellation) { m_enableEchoCancellation = enableEchoCancellation; }

    void addClient(CoreAudioCaptureSource&);
    void removeClient(CoreAudioCaptureSource&);
    void clearClients();

    virtual bool hasAudioUnit() const = 0;
    void setCaptureDevice(String&&, uint32_t, bool isDefault);

    virtual LongCapabilityRange sampleRateCapacities() const = 0;
    virtual int actualSampleRate() const { return sampleRate(); }

    bool isRenderingAudio() const { return m_isRenderingAudio; }
    bool hasClients() const { return !m_clients.isEmptyIgnoringNullReferences(); }

    const String& persistentIDForTesting() const { return m_capturingDevice ? m_capturingDevice->first : emptyString(); }

    void handleNewCurrentMicrophoneDevice(CaptureDevice&&);

    uint32_t captureDeviceID() const { return m_capturingDevice ? m_capturingDevice->second : 0; }

protected:
    BaseAudioSharedUnit();

    void forEachClient(const Function<void(CoreAudioCaptureSource&)>&) const;
    void captureFailed();
    void continueStartProducingData();

    virtual void cleanupAudioUnit() = 0;
    virtual OSStatus startInternal() = 0;
    virtual void stopInternal() = 0;
    virtual OSStatus reconfigureAudioUnit() = 0;
    virtual void resetSampleRate() = 0;
    virtual void captureDeviceChanged() = 0;

    void setSuspended(bool value) { m_suspended = value; }

    void audioSamplesAvailable(const MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t /*numberOfFrames*/);

    const String& persistentID() const { return m_capturingDevice ? m_capturingDevice->first : emptyString(); }

    void setIsRenderingAudio(bool);

protected:
    void setIsProducingMicrophoneSamples(bool);
    bool isProducingMicrophoneSamples() const { return m_isProducingMicrophoneSamples; }
    void setOutputDeviceID(uint32_t deviceID) { m_outputDeviceID = deviceID; }

    virtual void isProducingMicrophoneSamplesChanged() { }
    virtual void validateOutputDevice(uint32_t /* currentOutputDeviceID */) { }
    virtual bool migrateToNewDefaultDevice(const CaptureDevice&) { return false; }

    void setVoiceActivityListenerCallback(Function<void()>&& callback) { m_voiceActivityCallback = WTFMove(callback); }
    bool hasVoiceActivityListenerCallback() const { return !!m_voiceActivityCallback; }
    void voiceActivityDetected();

    void disableVoiceActivityThrottleTimerForTesting() { m_voiceActivityThrottleTimer.stop(); }
    void stopRunning();

    bool isCapturingWithDefaultMicrophone() const { return m_isCapturingWithDefaultMicrophone; }

private:
    OSStatus startUnit();
    bool shouldContinueRunning() const { return m_producingCount || m_isRenderingAudio || hasClients(); }

    virtual void willChangeCaptureDevice() { };

    // RealtimeMediaSourceCenterObserver
    void devicesChanged() final;
    void deviceWillBeRemoved(const String&) final { }

    bool m_enableEchoCancellation { true };
    double m_volume { 1 };
    int m_sampleRate;
    bool m_suspended { false };
    bool m_needsReconfiguration { false };
    bool m_isRenderingAudio { false };

    int32_t m_producingCount { 0 };

    uint32_t m_outputDeviceID { 0 };
    std::optional<std::pair<String, uint32_t>> m_capturingDevice;

    ThreadSafeWeakHashSet<CoreAudioCaptureSource> m_clients;
    Vector<ThreadSafeWeakPtr<CoreAudioCaptureSource>> m_audioThreadClients WTF_GUARDED_BY_LOCK(m_audioThreadClientsLock);
    Lock m_audioThreadClientsLock;

    bool m_isCapturingWithDefaultMicrophone { false };
    bool m_isProducingMicrophoneSamples { true };
    Function<void()> m_voiceActivityCallback;
    Timer m_voiceActivityThrottleTimer;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
