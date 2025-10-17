/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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
#include "config.h"
#include "BaseAudioSharedUnit.h"

#if ENABLE(MEDIA_STREAM)

#include "AudioSession.h"
#include "CaptureDeviceManager.h"
#include "CoreAudioCaptureSource.h"
#include "DeprecatedGlobalSettings.h"
#include "Logging.h"
#include "PlatformMediaSessionManager.h"

namespace WebCore {

constexpr Seconds voiceActivityThrottlingDuration = 5_s;

BaseAudioSharedUnit::BaseAudioSharedUnit()
    : m_sampleRate(AudioSession::protectedSharedSession()->sampleRate())
    , m_voiceActivityThrottleTimer([] { })
{
    RealtimeMediaSourceCenter::singleton().addDevicesChangedObserver(*this);
}

BaseAudioSharedUnit::~BaseAudioSharedUnit()
{
    RealtimeMediaSourceCenter::singleton().removeDevicesChangedObserver(*this);
}

void BaseAudioSharedUnit::addClient(CoreAudioCaptureSource& client)
{
    ASSERT(isMainThread());
    m_clients.add(client);
    Locker locker { m_audioThreadClientsLock };
    m_audioThreadClients = m_clients.weakValues();
}

void BaseAudioSharedUnit::removeClient(CoreAudioCaptureSource& client)
{
    ASSERT(isMainThread());
    m_clients.remove(client);
    {
        Locker locker { m_audioThreadClientsLock };
        m_audioThreadClients = m_clients.weakValues();
    }

    if (!shouldContinueRunning())
        stopRunning();
}

void BaseAudioSharedUnit::clearClients()
{
    ASSERT(isMainThread());
    m_clients.clear();
    Locker locker { m_audioThreadClientsLock };
    m_audioThreadClients.clear();
}

void BaseAudioSharedUnit::forEachClient(const Function<void(CoreAudioCaptureSource&)>& apply) const
{
    ASSERT(isMainThread());
    m_clients.forEach(apply);
}

const static OSStatus lowPriorityError1 = 560557684;
const static OSStatus lowPriorityError2 = 561017449;
void BaseAudioSharedUnit::startProducingData()
{
    ASSERT(isMainThread());

    if (m_suspended)
        resume();

    setIsProducingMicrophoneSamples(true);

    if (++m_producingCount != 1)
        return;

#if PLATFORM(MAC)
    prewarmAudioUnitCreation([weakThis = WeakPtr { *this }] {
        if (RefPtr protectedThis = weakThis.get())
            protectedThis->continueStartProducingData();
    });
#else
    continueStartProducingData();
#endif
}

void BaseAudioSharedUnit::continueStartProducingData()
{
    if (!m_producingCount)
        return;

    if (isProducingData())
        return;

    if (hasAudioUnit()) {
        cleanupAudioUnit();
        ASSERT(!hasAudioUnit());
    }
    auto error = startUnit();
    if (error) {
        if (error == lowPriorityError1 || error == lowPriorityError2) {
            RELEASE_LOG_ERROR(WebRTC, "BaseAudioSharedUnit::startProducingData failed due to not high enough priority, suspending unit");
            suspend();
        } else
            captureFailed();
    }
}

OSStatus BaseAudioSharedUnit::startUnit()
{
    forEachClient([](auto& client) {
        client.audioUnitWillStart();
    });
    ASSERT(!DeprecatedGlobalSettings::shouldManageAudioSessionCategory() || AudioSession::sharedSession().category() == AudioSession::CategoryType::PlayAndRecord);

#if PLATFORM(IOS_FAMILY)
    if (AudioSession::sharedSession().category() != AudioSession::CategoryType::PlayAndRecord) {
        RELEASE_LOG_ERROR(WebRTC, "BaseAudioSharedUnit::startUnit cannot call startInternal if category is not set to PlayAndRecord");
        return lowPriorityError2;
    }
#endif

    return startInternal();
}

void BaseAudioSharedUnit::prepareForNewCapture()
{
    m_volume = 1;
    resetSampleRate();

    if (!m_suspended)
        return;
    m_suspended = false;

    if (!m_producingCount)
        return;

    RELEASE_LOG_ERROR(WebRTC, "BaseAudioSharedUnit::prepareForNewCapture, notifying suspended sources of capture failure");
    captureFailed();
}

void BaseAudioSharedUnit::setCaptureDevice(String&& persistentID, uint32_t captureDeviceID, bool isDefault)
{
    bool hasChanged = this->persistentID() != persistentID || this->captureDeviceID() != captureDeviceID || m_isCapturingWithDefaultMicrophone != isDefault;
    if (hasChanged)
        willChangeCaptureDevice();

    m_capturingDevice = { WTFMove(persistentID), captureDeviceID };
    m_isCapturingWithDefaultMicrophone = isDefault;

    if (hasChanged)
        captureDeviceChanged();
}

void BaseAudioSharedUnit::devicesChanged()
{
    Ref protectedThis { *this };

    if (!hasAudioUnit())
        return;

    auto devices = RealtimeMediaSourceCenter::singleton().audioCaptureFactory().audioCaptureDeviceManager().captureDevices();
    auto persistentID = this->persistentID();
    if (persistentID.isEmpty())
        return;

    if (WTF::anyOf(devices, [&persistentID] (auto& device) { return persistentID == device.persistentId(); })) {
        validateOutputDevice(m_outputDeviceID);
        return;
    }

    if (devices.size() && m_isCapturingWithDefaultMicrophone && migrateToNewDefaultDevice(devices[0])) {
        RELEASE_LOG_ERROR(WebRTC, "BaseAudioSharedUnit::devicesChanged - migrating to new default device");
        return;
    }

    RELEASE_LOG_ERROR(WebRTC, "BaseAudioSharedUnit::devicesChanged - failing capture, capturing device is missing");
    captureFailed();
}

void BaseAudioSharedUnit::captureFailed()
{
    RELEASE_LOG_ERROR(WebRTC, "BaseAudioSharedUnit::captureFailed");
    forEachClient([](auto& client) {
        client.captureFailed();
    });

    m_producingCount = 0;

    clearClients();

    stopRunning();
}

void BaseAudioSharedUnit::stopProducingData()
{
    ASSERT(isMainThread());
    ASSERT(m_producingCount);

    if (m_producingCount && --m_producingCount)
        return;

    if (shouldContinueRunning()) {
        setIsProducingMicrophoneSamples(false);
        return;
    }

    stopRunning();
}

void BaseAudioSharedUnit::setIsProducingMicrophoneSamples(bool value)
{
    m_isProducingMicrophoneSamples = value;
    isProducingMicrophoneSamplesChanged();
}

void BaseAudioSharedUnit::setIsRenderingAudio(bool value)
{
    m_isRenderingAudio = value;
    if (!shouldContinueRunning())
        stopRunning();
}

void BaseAudioSharedUnit::stopRunning()
{
    stopInternal();
    cleanupAudioUnit();
}

void BaseAudioSharedUnit::reconfigure()
{
    ASSERT(isMainThread());
    if (m_suspended) {
        m_needsReconfiguration = true;
        return;
    }
    reconfigureAudioUnit();
}

OSStatus BaseAudioSharedUnit::resume()
{
    ASSERT(isMainThread());
    if (!m_suspended)
        return 0;

    ASSERT(!isProducingData());

    RELEASE_LOG_INFO(WebRTC, "BaseAudioSharedUnit::resume");

    m_suspended = false;

    if (m_needsReconfiguration) {
        m_needsReconfiguration = false;
        reconfigure();
    }

    ASSERT(!m_producingCount);

    callOnMainThread([weakThis = WeakPtr { this }] {
        RefPtr protectedThis = weakThis.get();
        if (!protectedThis || protectedThis->m_suspended)
            return;

        protectedThis->forEachClient([](auto& client) {
            if (client.canResumeAfterInterruption())
                client.setMuted(false);
        });
    });

    return 0;
}

OSStatus BaseAudioSharedUnit::suspend()
{
    ASSERT(isMainThread());

    RELEASE_LOG_INFO(WebRTC, "BaseAudioSharedUnit::suspend");

    m_suspended = true;
    stopInternal();

    forEachClient([](auto& client) {
        client.setCanResumeAfterInterruption(client.isProducingData());
        client.setMuted(true);
    });

    ASSERT(!m_producingCount);

    return 0;
}

void BaseAudioSharedUnit::audioSamplesAvailable(const MediaTime& time, const PlatformAudioData& data, const AudioStreamDescription& description, size_t numberOfFrames)
{
    // We hold the lock here since adding/removing clients can only happen in main thread.
    Locker locker { m_audioThreadClientsLock };

    // For performance reasons, we forbid heap allocations while doing rendering on the capture audio thread.
    ForbidMallocUseForCurrentThreadScope forbidMallocUse;

    for (auto& weakClient : m_audioThreadClients) {
        RefPtr client = weakClient.get();
        if (!client)
            continue;
        if (client->isProducingData())
            client->audioSamplesAvailable(time, data, description, numberOfFrames);
    }
}

void BaseAudioSharedUnit::handleNewCurrentMicrophoneDevice(CaptureDevice&& device)
{
    forEachClient([&device](auto& client) {
        client.handleNewCurrentMicrophoneDevice(device);
    });
}

void BaseAudioSharedUnit::voiceActivityDetected()
{
    if (m_voiceActivityThrottleTimer.isActive() || !m_voiceActivityCallback)
        return;

    RELEASE_LOG_INFO(WebRTC, "BaseAudioSharedUnit::voiceActivityDetected");

    m_voiceActivityCallback();
    m_voiceActivityThrottleTimer.startOneShot(voiceActivityThrottlingDuration);
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
