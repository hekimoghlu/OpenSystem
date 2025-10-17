/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
#include "MockRealtimeAudioSource.h"

#if ENABLE(MEDIA_STREAM)
#include "AudioSession.h"
#include "CaptureDevice.h"
#include "Logging.h"
#include "MediaConstraints.h"
#include "MockRealtimeMediaSourceCenter.h"
#include "NotImplemented.h"
#include "PlatformMediaSessionManager.h"
#include "RealtimeMediaSourceSettings.h"
#include <wtf/UUID.h>

#if PLATFORM(COCOA)
#include "MockAudioSharedUnit.h"
#endif

#if USE(GSTREAMER)
#include "MockRealtimeAudioSourceGStreamer.h"
#endif

namespace WebCore {

#if !PLATFORM(MAC) && !PLATFORM(IOS_FAMILY) && !USE(GSTREAMER)
CaptureSourceOrError MockRealtimeAudioSource::create(String&& deviceID, String&& name, MediaDeviceHashSalts&& hashSalts, const MediaConstraints* constraints, std::optional<PageIdentifier>)
{
#ifndef NDEBUG
    auto device = MockRealtimeMediaSourceCenter::mockDeviceWithPersistentID(deviceID);
    ASSERT(device);
    if (!device)
        return { "No mock microphone device"_s };
#endif

    auto source = adoptRef(*new MockRealtimeAudioSource(WTFMove(deviceID), WTFMove(name), WTFMove(hashSalts)));
    if (constraints) {
        if (auto error = source->applyConstraints(*constraints))
            return CaptureSourceOrError({ WTFMove(error->invalidConstraint), MediaAccessDenialReason::InvalidConstraint });
    }

    return CaptureSourceOrError(WTFMove(source));
}
#endif

MockRealtimeAudioSource::MockRealtimeAudioSource(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&& hashSalts, std::optional<PageIdentifier> pageIdentifier)
    : RealtimeMediaSource(CaptureDevice { WTFMove(deviceID), CaptureDevice::DeviceType::Microphone, WTFMove(name) }, WTFMove(hashSalts), pageIdentifier)
    , m_workQueue(WorkQueue::create("MockRealtimeAudioSource Render Queue"_s))
    , m_timer(RunLoop::current(), this, &MockRealtimeAudioSource::tick)
{
    auto device = MockRealtimeMediaSourceCenter::mockDeviceWithPersistentID(persistentID());
    ASSERT(device);
    m_device = *device;

    setSampleRate(std::get<MockMicrophoneProperties>(m_device.properties).defaultSampleRate);
    initializeEchoCancellation(std::get<MockMicrophoneProperties>(m_device.properties).echoCancellation.value_or(true));
}

MockRealtimeAudioSource::~MockRealtimeAudioSource()
{
}

const RealtimeMediaSourceSettings& MockRealtimeAudioSource::settings()
{
    if (!m_currentSettings) {
        RealtimeMediaSourceSettings settings;
        settings.setDeviceId(hashedId());
        settings.setGroupId(hashedGroupId());
        settings.setVolume(volume());
        settings.setEchoCancellation(echoCancellation());
        settings.setSampleRate(sampleRate());
        settings.setLabel(AtomString { name() });

        RealtimeMediaSourceSupportedConstraints supportedConstraints;
        supportedConstraints.setSupportsDeviceId(true);
        supportedConstraints.setSupportsGroupId(true);
        supportedConstraints.setSupportsVolume(true);
        supportedConstraints.setSupportsEchoCancellation(true);
        supportedConstraints.setSupportsSampleRate(true);
        settings.setSupportedConstraints(supportedConstraints);

        m_currentSettings = WTFMove(settings);
    }
    return m_currentSettings.value();
}

void MockRealtimeAudioSource::setChannelCount(unsigned channelCount)
{
    if (channelCount > 2)
        return;

    m_channelCount = channelCount;
    settingsDidChange(RealtimeMediaSourceSettings::Flag::SampleRate);
}

const RealtimeMediaSourceCapabilities& MockRealtimeAudioSource::capabilities()
{
    if (!m_capabilities) {
        RealtimeMediaSourceCapabilities capabilities(settings().supportedConstraints());

        capabilities.setDeviceId(hashedId());
        capabilities.setGroupId(hashedGroupId());
        capabilities.setVolume({ 0.0, 1.0 });

        auto echoCancellation = std::get<MockMicrophoneProperties>(m_device.properties).echoCancellation;
        capabilities.setEchoCancellation(echoCancellation ? (*echoCancellation ? RealtimeMediaSourceCapabilities::EchoCancellation::On : RealtimeMediaSourceCapabilities::EchoCancellation::Off) : RealtimeMediaSourceCapabilities::EchoCancellation::OnOrOff);
        capabilities.setSampleRate({ 44100, 96000 });

        m_capabilities = WTFMove(capabilities);
    }
    return m_capabilities.value();
}

void MockRealtimeAudioSource::settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag>)
{
    m_currentSettings = std::nullopt;
}

void MockRealtimeAudioSource::startProducingData()
{
#if PLATFORM(IOS_FAMILY)
    PlatformMediaSessionManager::singleton().sessionCanProduceAudioChanged();
    ASSERT(AudioSession::sharedSession().category() == AudioSession::CategoryType::PlayAndRecord);
    ASSERT(AudioSession::sharedSession().mode() == AudioSession::Mode::VideoChat);
#endif

    if (!sampleRate())
        setSampleRate(std::get<MockMicrophoneProperties>(m_device.properties).defaultSampleRate);

    m_startTime = MonotonicTime::now();
    m_timer.startRepeating(renderInterval());
}

void MockRealtimeAudioSource::stopProducingData()
{
    m_timer.stop();
    m_startTime = MonotonicTime::nan();
}

void MockRealtimeAudioSource::tick()
{
    if (m_lastRenderTime.isNaN())
        m_lastRenderTime = MonotonicTime::now();

    MonotonicTime now = MonotonicTime::now();

    if (m_delayUntil) {
        if (m_delayUntil < now)
            return;
        m_delayUntil = MonotonicTime();
    }

    Seconds delta = now - m_lastRenderTime;
    m_lastRenderTime = now;

    m_workQueue->dispatch([this, delta, protectedThis = Ref { *this }] {
        render(delta);
    });
}

void MockRealtimeAudioSource::delaySamples(Seconds delta)
{
    m_delayUntil = MonotonicTime::now() + delta;
}

void MockRealtimeAudioSource::setIsInterrupted(bool isInterrupted)
{
    UNUSED_PARAM(isInterrupted);
#if PLATFORM(COCOA)
    if (isInterrupted)
        CoreAudioSharedUnit::singleton().suspend();
    else
        CoreAudioSharedUnit::singleton().resume();
#elif USE(GSTREAMER)
    for (auto* source : MockRealtimeAudioSourceGStreamer::allMockRealtimeAudioSources())
        source->setInterruptedForTesting(isInterrupted);
#endif
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
