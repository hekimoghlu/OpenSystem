/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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
#include "CaptureDevice.h"
#include "GStreamerAudioCapturer.h"
#include "GStreamerCaptureDevice.h"
#include "RealtimeMediaSource.h"

namespace WebCore {

class GStreamerAudioCaptureSource : public RealtimeMediaSource, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<GStreamerAudioCaptureSource>,  public GStreamerCapturerObserver {
public:
    static CaptureSourceOrError create(String&& deviceID, MediaDeviceHashSalts&&, const MediaConstraints*);
    WEBCORE_EXPORT static AudioCaptureFactory& factory();

    const RealtimeMediaSourceCapabilities& capabilities() override;
    const RealtimeMediaSourceSettings& settings() override;

    GstElement* pipeline() { return m_capturer->pipeline(); }
    GStreamerCapturer* capturer() { return m_capturer.get(); }

    std::pair<GstClockTime, GstClockTime> queryCaptureLatency() const final;

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;
    virtual ~GStreamerAudioCaptureSource();

    // GStreamerCapturerObserver
    void captureEnded()final;

protected:
    GStreamerAudioCaptureSource(GStreamerCaptureDevice&&, MediaDeviceHashSalts&&);
    void startProducingData() override;
    void stopProducingData() override;
    CaptureDevice::DeviceType deviceType() const override { return CaptureDevice::DeviceType::Microphone; }

    mutable std::optional<RealtimeMediaSourceCapabilities> m_capabilities;
    mutable std::optional<RealtimeMediaSourceSettings> m_currentSettings;

private:
    bool interrupted() const final;
    void setInterruptedForTesting(bool) final;

    bool isCaptureSource() const final { return true; }
    void settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag>) final;

    RefPtr<GStreamerAudioCapturer> m_capturer;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
