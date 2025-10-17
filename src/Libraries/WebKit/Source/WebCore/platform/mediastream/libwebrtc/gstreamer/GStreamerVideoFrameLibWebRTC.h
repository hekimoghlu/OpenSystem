/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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

#if USE(GSTREAMER) && USE(LIBWEBRTC)

#include "GStreamerCommon.h"
#include "LibWebRTCMacros.h"
#include "webrtc/api/video/i420_buffer.h"
#include "webrtc/api/video/video_frame.h"
#include "webrtc/common_video/include/video_frame_buffer.h"
#include "webrtc/rtc_base/ref_counted_object.h"

namespace WebCore {

WARN_UNUSED_RETURN GRefPtr<GstSample> convertLibWebRTCVideoFrameToGStreamerSample(const webrtc::VideoFrame&);

webrtc::VideoFrame convertGStreamerSampleToLibWebRTCVideoFrame(GRefPtr<GstSample>&&, uint32_t rtpTimestamp);

class GStreamerVideoFrameLibWebRTC : public rtc::RefCountedObject<webrtc::VideoFrameBuffer> {
public:
    GStreamerVideoFrameLibWebRTC(GRefPtr<GstSample>&& sample, GstVideoInfo info)
        : m_sample(WTFMove(sample))
        , m_info(info)
    {
    }

    static rtc::scoped_refptr<webrtc::VideoFrameBuffer> create(GRefPtr<GstSample>&&);

    GstSample* getSample() const { return m_sample.get(); }
    rtc::scoped_refptr<webrtc::I420BufferInterface> ToI420() final;

    int width() const final { return GST_VIDEO_INFO_WIDTH(&m_info); }
    int height() const final { return GST_VIDEO_INFO_HEIGHT(&m_info); }

private:
    webrtc::VideoFrameBuffer::Type type() const final { return Type::kNative; }

    GRefPtr<GstSample> m_sample;
    GstVideoInfo m_info;
};

}

#endif // USE(GSTREAMER) && USE(LIBWEBRTC)
