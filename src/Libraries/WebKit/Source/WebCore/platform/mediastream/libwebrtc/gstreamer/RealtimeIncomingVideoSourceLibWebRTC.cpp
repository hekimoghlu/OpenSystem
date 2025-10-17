/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
#include "RealtimeIncomingVideoSourceLibWebRTC.h"

#include "GStreamerVideoFrameLibWebRTC.h"
#include "VideoFrameGStreamer.h"

namespace WebCore {

Ref<RealtimeIncomingVideoSource> RealtimeIncomingVideoSource::create(rtc::scoped_refptr<webrtc::VideoTrackInterface>&& videoTrack, String&& trackId)
{
    auto source = RealtimeIncomingVideoSourceLibWebRTC::create(WTFMove(videoTrack), WTFMove(trackId));
    source->start();
    return source;
}

Ref<RealtimeIncomingVideoSourceLibWebRTC> RealtimeIncomingVideoSourceLibWebRTC::create(rtc::scoped_refptr<webrtc::VideoTrackInterface>&& videoTrack, String&& trackId)
{
    return adoptRef(*new RealtimeIncomingVideoSourceLibWebRTC(WTFMove(videoTrack), WTFMove(trackId)));
}

RealtimeIncomingVideoSourceLibWebRTC::RealtimeIncomingVideoSourceLibWebRTC(rtc::scoped_refptr<webrtc::VideoTrackInterface>&& videoTrack, String&& videoTrackId)
    : RealtimeIncomingVideoSource(WTFMove(videoTrack), WTFMove(videoTrackId))
{
}

void RealtimeIncomingVideoSourceLibWebRTC::OnFrame(const webrtc::VideoFrame& frame)
{
    if (!isProducingData())
        return;

    auto presentationTime = MediaTime(frame.timestamp_us(), G_USEC_PER_SEC);
    if (frame.video_frame_buffer()->type() == webrtc::VideoFrameBuffer::Type::kNative) {
        auto* framebuffer = static_cast<GStreamerVideoFrameLibWebRTC*>(frame.video_frame_buffer().get());
        videoFrameAvailable(VideoFrameGStreamer::createWrappedSample(framebuffer->getSample(), presentationTime, static_cast<VideoFrame::Rotation>(frame.rotation())), { });
    } else {
        auto gstSample = convertLibWebRTCVideoFrameToGStreamerSample(frame);
        auto metadata = std::make_optional(metadataFromVideoFrame(frame));
        videoFrameAvailable(VideoFrameGStreamer::create(WTFMove(gstSample), { }, presentationTime, static_cast<VideoFrame::Rotation>(frame.rotation()), false, WTFMove(metadata)), { });
    }
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)
