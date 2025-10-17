/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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

#include "MockRealtimeVideoSourceGStreamer.h"

namespace WebCore {

class MockDisplayCaptureSourceGStreamer : public RealtimeVideoCaptureSource, RealtimeMediaSource::VideoFrameObserver {
public:
    static CaptureSourceOrError create(const CaptureDevice&, MediaDeviceHashSalts&&, const MediaConstraints*, std::optional<PageIdentifier>);

    void requestToEnd(RealtimeMediaSourceObserver&) final;
    bool isProducingData() const final { return m_source->isProducingData(); }
    void setMuted(bool isMuted) final;
    const IntSize size() const final { return m_source->size(); }

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "MockDisplayCaptureSourceGStreamer"_s; }
#endif

protected:
    // RealtimeMediaSource::VideoFrameObserver
    void videoFrameAvailable(VideoFrame&, VideoFrameTimeMetadata) final;

    void generatePresets() override { };
    const Vector<VideoPreset>& presets() final { return m_presets; }

private:
    MockDisplayCaptureSourceGStreamer(const CaptureDevice&, Ref<MockRealtimeVideoSourceGStreamer>&&, MediaDeviceHashSalts&&, std::optional<PageIdentifier>);
    ~MockDisplayCaptureSourceGStreamer();

    void startProducingData() final { m_source->start(); }
    void stopProducingData() final;
    void settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag>) final { m_currentSettings = { }; }
    bool isCaptureSource() const final { return true; }
    const RealtimeMediaSourceCapabilities& capabilities() final;
    const RealtimeMediaSourceSettings& settings() final;
    CaptureDevice::DeviceType deviceType() const final { return m_deviceType; }

    Vector<VideoPreset> m_presets;
    Ref<MockRealtimeVideoSourceGStreamer> m_source;
    CaptureDevice::DeviceType m_deviceType;
    std::optional<RealtimeMediaSourceCapabilities> m_capabilities;
    std::optional<RealtimeMediaSourceSettings> m_currentSettings;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
