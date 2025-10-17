/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 24, 2023.
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

#if ENABLE(VIDEO) && USE(GSTREAMER)

#include "VideoFrame.h"
#include "VideoFrameMetadataGStreamer.h"
#include <gst/video/video-format.h>
#include <wtf/glib/GRefPtr.h>

typedef struct _GstSample GstSample;

namespace WebCore {

class PixelBuffer;
class IntSize;
class ImageGStreamer;

class VideoFrameGStreamer final : public VideoFrame {
public:

    enum class CanvasContentType {
        WebGL,
        Canvas2D
    };

    static Ref<VideoFrameGStreamer> create(GRefPtr<GstSample>&&, const IntSize& presentationSize, const MediaTime& presentationTime = MediaTime::invalidTime(), Rotation videoRotation = Rotation::None, bool videoMirrored = false, std::optional<VideoFrameTimeMetadata>&& = std::nullopt, std::optional<PlatformVideoColorSpace>&& = std::nullopt);

    static Ref<VideoFrameGStreamer> createWrappedSample(const GRefPtr<GstSample>&, const MediaTime& presentationTime = MediaTime::invalidTime(), Rotation videoRotation = Rotation::None);

    static RefPtr<VideoFrameGStreamer> createFromPixelBuffer(Ref<PixelBuffer>&&, CanvasContentType canvasContentType, Rotation videoRotation = VideoFrame::Rotation::None, const MediaTime& presentationTime = MediaTime::invalidTime(), const IntSize& destinationSize = { }, double frameRate = 1, bool videoMirrored = false, std::optional<VideoFrameTimeMetadata>&& metadata = std::nullopt, PlatformVideoColorSpace&& = { });

    void setFrameRate(double);
    void setMaxFrameRate(double);

    void setPresentationTime(const MediaTime&);

    RefPtr<VideoFrameGStreamer> resizeTo(const IntSize&);

    GRefPtr<GstSample> resizedSample(const IntSize&);

    GRefPtr<GstSample> downloadSample(std::optional<GstVideoFormat> = { });

    GstSample* sample() const { return m_sample.get(); }

    RefPtr<ImageGStreamer> convertToImage();

    IntSize presentationSize() const final { return m_presentationSize; }
    uint32_t pixelFormat() const final;

private:
    VideoFrameGStreamer(GRefPtr<GstSample>&&, const IntSize& presentationSize, const MediaTime& presentationTime = MediaTime::invalidTime(), Rotation = Rotation::None, bool videoMirrored = false, std::optional<VideoFrameTimeMetadata>&& = std::nullopt, PlatformVideoColorSpace&& = { });
    VideoFrameGStreamer(const GRefPtr<GstSample>&, const IntSize& presentationSize, const MediaTime& presentationTime = MediaTime::invalidTime(), Rotation = Rotation::None, PlatformVideoColorSpace&& = { });

    bool isGStreamer() const final { return true; }
    Ref<VideoFrame> clone() final;

    GRefPtr<GstSample> convert(GstVideoFormat, const IntSize&);

    GRefPtr<GstSample> m_sample;
    IntSize m_presentationSize;
    mutable GstVideoFormat m_cachedVideoFormat { GST_VIDEO_FORMAT_UNKNOWN };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::VideoFrameGStreamer)
static bool isType(const WebCore::VideoFrame& frame) { return frame.isGStreamer(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
