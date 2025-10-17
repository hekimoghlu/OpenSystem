/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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

#if ENABLE(MEDIA_STREAM) && USE(GSTREAMER)

#include "GStreamerAudioCapturer.h"
#include "GStreamerAudioData.h"
#include "GStreamerAudioStreamDescription.h"
#include "MockRealtimeAudioSource.h"

namespace WebCore {

class MockRealtimeAudioSourceGStreamer final : public MockRealtimeAudioSource, GStreamerCapturerObserver {
public:
    static Ref<MockRealtimeAudioSource> createForMockAudioCapturer(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&&);

    static const UncheckedKeyHashSet<MockRealtimeAudioSource*>& allMockRealtimeAudioSources();

    ~MockRealtimeAudioSourceGStreamer();

    // GStreamerCapturerObserver
    void captureEnded() final;

    std::pair<GstClockTime, GstClockTime> queryCaptureLatency() const final;

protected:
    void render(Seconds) final;
    void settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag>) final;

private:
    friend class MockRealtimeAudioSource;
    MockRealtimeAudioSourceGStreamer(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&&);
    void reconfigure();

    void startProducingData() final;
    void stopProducingData() final;

    bool interrupted() const final { return m_isInterrupted; };
    void setInterruptedForTesting(bool) final;

    std::optional<GStreamerAudioStreamDescription> m_streamFormat;
    GRefPtr<GstCaps> m_caps;
    Vector<float> m_bipBopBuffer;
    uint32_t m_maximiumFrameCount;
    uint64_t m_samplesEmitted { 0 };
    uint64_t m_samplesRendered { 0 };
    bool m_isInterrupted { false };
    RefPtr<GStreamerAudioCapturer> m_capturer;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
