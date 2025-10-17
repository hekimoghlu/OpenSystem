/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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

#include "ImageBuffer.h"
#include "MockMediaDevice.h"
#include "RealtimeMediaSourceFactory.h"
#include <wtf/RunLoop.h>
#include <wtf/WorkQueue.h>

namespace WebCore {

class MockRealtimeAudioSource : public RealtimeMediaSource, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<MockRealtimeAudioSource, WTF::DestructionThread::MainRunLoop> {
public:
    static CaptureSourceOrError create(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&&, const MediaConstraints*, std::optional<PageIdentifier>);
    virtual ~MockRealtimeAudioSource();

    static void setIsInterrupted(bool);

    WEBCORE_EXPORT void setChannelCount(unsigned);

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

protected:
    MockRealtimeAudioSource(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&&, std::optional<PageIdentifier>);

    virtual void render(Seconds) = 0;
    void settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag>) override;

    static Seconds renderInterval() { return 60_ms; }

private:
    friend class MockRealtimeAudioSourceGStreamer;

    const RealtimeMediaSourceCapabilities& capabilities() final;
    const RealtimeMediaSourceSettings& settings() final;

    void startProducingData() override;
    void stopProducingData() override;

    bool isCaptureSource() const final { return true; }
    CaptureDevice::DeviceType deviceType() const final { return CaptureDevice::DeviceType::Microphone; }

    void delaySamples(Seconds) final;
    bool isMockSource() const final { return true; }

    void tick();

protected:
    Ref<WorkQueue> m_workQueue;
    unsigned m_channelCount { 2 };

private:
    std::optional<RealtimeMediaSourceCapabilities> m_capabilities;
    std::optional<RealtimeMediaSourceSettings> m_currentSettings;
    RealtimeMediaSourceSupportedConstraints m_supportedConstraints;

    RunLoop::Timer m_timer;
    MonotonicTime m_startTime { MonotonicTime::nan() };
    MonotonicTime m_lastRenderTime { MonotonicTime::nan() };
    Seconds m_elapsedTime { 0_s };
    MonotonicTime m_delayUntil;
    MockMediaDevice m_device;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::MockRealtimeAudioSource)
    static bool isType(const WebCore::RealtimeMediaSource& source) { return source.isCaptureSource() && source.isMockSource() && source.deviceType() == WebCore::CaptureDevice::DeviceType::Microphone; }
SPECIALIZE_TYPE_TRAITS_END()


#endif // ENABLE(MEDIA_STREAM)
