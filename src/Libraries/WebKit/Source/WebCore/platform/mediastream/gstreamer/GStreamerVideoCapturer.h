/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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

#include "GStreamerCapturer.h"

namespace WebCore {

class VideoFrameGStreamer;

class GStreamerVideoCapturer final : public GStreamerCapturer {
    friend class GStreamerVideoCaptureSource;
    friend class MockRealtimeVideoSourceGStreamer;
public:
    GStreamerVideoCapturer(GStreamerCaptureDevice&&);
    GStreamerVideoCapturer(const char* sourceFactory, CaptureDevice::DeviceType);
    ~GStreamerVideoCapturer() = default;

    GstElement* createSource() final;
    GstElement* createConverter() final;
    const char* name() final { return "Video"; }

    using NodeAndFD = std::pair<uint32_t, int>;

    using SinkVideoFrameCallback = Function<void(Ref<VideoFrameGStreamer>&&)>;
    void setSinkVideoFrameCallback(SinkVideoFrameCallback&&);

private:
    bool setSize(const IntSize&);
    const IntSize& size() const { return m_size; }
    bool setFrameRate(double);
    void reconfigure();

    GstVideoInfo getBestFormat();

    void setPipewireNodeAndFD(const NodeAndFD& nodeAndFd) { m_nodeAndFd = nodeAndFd; }
    bool isCapturingDisplay() const { return m_nodeAndFd.has_value(); }

    std::optional<NodeAndFD> m_nodeAndFd;
    GRefPtr<GstElement> m_videoSrcMIMETypeFilter;
    std::pair<unsigned long, SinkVideoFrameCallback> m_sinkVideoFrameCallback;
    IntSize m_size;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
