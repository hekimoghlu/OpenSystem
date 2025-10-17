/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#include "GStreamerVideoCapturer.h"
#include "RealtimeVideoCaptureSource.h"
#include "VideoFrameGStreamer.h"

namespace WebCore {

using NodeAndFD = GStreamerVideoCapturer::NodeAndFD;

class GStreamerVideoCaptureSource : public RealtimeVideoCaptureSource, GStreamerCapturerObserver {
public:
    static CaptureSourceOrError create(String&& deviceID, MediaDeviceHashSalts&&, const MediaConstraints*);
    static CaptureSourceOrError createPipewireSource(String&& deviceID, const NodeAndFD&, MediaDeviceHashSalts&&, const MediaConstraints*, CaptureDevice::DeviceType);

    WEBCORE_EXPORT static VideoCaptureFactory& factory();

    WEBCORE_EXPORT static DisplayCaptureFactory& displayFactory();

    const RealtimeMediaSourceCapabilities& capabilities() override;
    const RealtimeMediaSourceSettings& settings() override;
    GstElement* pipeline() { return m_capturer->pipeline(); }
    GStreamerCapturer* capturer() { return m_capturer.get(); }

    // GStreamerCapturerObserver
    void sourceCapsChanged(const GstCaps*) final;
    void captureEnded() final;

    std::pair<GstClockTime, GstClockTime> queryCaptureLatency() const final;

protected:
    GStreamerVideoCaptureSource(String&& deviceID, AtomString&& name, MediaDeviceHashSalts&&, const gchar* source_factory, CaptureDevice::DeviceType, const NodeAndFD&);
    GStreamerVideoCaptureSource(GStreamerCaptureDevice&&, MediaDeviceHashSalts&&);
    virtual ~GStreamerVideoCaptureSource();
    void startProducingData() override;
    void stopProducingData() override;
    bool canResizeVideoFrames() const final { return true; }
    void generatePresets() override;
    void setSizeFrameRateAndZoom(const VideoPresetConstraints&) override;

    mutable std::optional<RealtimeMediaSourceCapabilities> m_capabilities;
    mutable std::optional<RealtimeMediaSourceSettings> m_currentSettings;
    CaptureDevice::DeviceType deviceType() const override { return m_deviceType; }

private:
    bool isCaptureSource() const final { return true; }
    void settingsDidChange(OptionSet<RealtimeMediaSourceSettings::Flag>) final;

    RefPtr<GStreamerVideoCapturer> m_capturer;
    CaptureDevice::DeviceType m_deviceType;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
