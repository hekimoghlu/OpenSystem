/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 28, 2022.
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

#if USE(LIBWEBRTC) && USE(GSTREAMER)
#include "RealtimeOutgoingVideoSourceLibWebRTC.h"

#include "GStreamerVideoFrameLibWebRTC.h"
#include "VideoFrameGStreamer.h"

namespace WebCore {

Ref<RealtimeOutgoingVideoSource> RealtimeOutgoingVideoSource::create(Ref<MediaStreamTrackPrivate>&& videoSource)
{
    return RealtimeOutgoingVideoSourceLibWebRTC::create(WTFMove(videoSource));
}

Ref<RealtimeOutgoingVideoSourceLibWebRTC> RealtimeOutgoingVideoSourceLibWebRTC::create(Ref<MediaStreamTrackPrivate>&& videoSource)
{
    return adoptRef(*new RealtimeOutgoingVideoSourceLibWebRTC(WTFMove(videoSource)));
}

RealtimeOutgoingVideoSourceLibWebRTC::RealtimeOutgoingVideoSourceLibWebRTC(Ref<MediaStreamTrackPrivate>&& videoSource)
    : RealtimeOutgoingVideoSource(WTFMove(videoSource))
{
}

void RealtimeOutgoingVideoSourceLibWebRTC::videoFrameAvailable(VideoFrame& videoFrame, VideoFrameTimeMetadata)
{
    switch (videoFrame.rotation()) {
    case VideoFrame::Rotation::None:
        m_currentRotation = webrtc::kVideoRotation_0;
        break;
    case VideoFrame::Rotation::UpsideDown:
        m_currentRotation = webrtc::kVideoRotation_180;
        break;
    case VideoFrame::Rotation::Right:
        m_currentRotation = webrtc::kVideoRotation_90;
        break;
    case VideoFrame::Rotation::Left:
        m_currentRotation = webrtc::kVideoRotation_270;
        break;
    }

    auto frameBuffer = GStreamerVideoFrameLibWebRTC::create(static_cast<VideoFrameGStreamer&>(videoFrame).sample());

    sendFrame(WTFMove(frameBuffer));
}

rtc::scoped_refptr<webrtc::VideoFrameBuffer> RealtimeOutgoingVideoSourceLibWebRTC::createBlackFrame(size_t  width, size_t  height)
{
    GstVideoInfo info;

    gst_video_info_set_format(&info, GST_VIDEO_FORMAT_RGB, width, height);

    GRefPtr<GstBuffer> buffer = adoptGRef(gst_buffer_new_allocate(nullptr, info.size, nullptr));
    GRefPtr<GstCaps> caps = adoptGRef(gst_video_info_to_caps(&info));

    GstMappedBuffer map(buffer.get(), GST_MAP_WRITE);
    memset(map.data(), 0, info.size);

    return GStreamerVideoFrameLibWebRTC::create(gst_sample_new(buffer.get(), caps.get(), NULL, NULL));
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)
