/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
#include "RealtimeOutgoingVideoSourceCocoa.h"

#if USE(LIBWEBRTC)

#include "CVUtilities.h"
#include "ImageRotationSessionVT.h"
#include "Logging.h"
#include "RealtimeIncomingVideoSourceCocoa.h"
#include "RealtimeVideoUtilities.h"
#include "VideoFrameLibWebRTC.h"

ALLOW_UNUSED_PARAMETERS_BEGIN

#include <webrtc/api/video/i420_buffer.h>
#include <webrtc/common_video/libyuv/include/webrtc_libyuv.h>
#include <webrtc/webkit_sdk/WebKit/WebKitUtilities.h>

ALLOW_UNUSED_PARAMETERS_END

#include <pal/cf/CoreMediaSoftLink.h>
#include "CoreVideoSoftLink.h"

namespace WebCore {

Ref<RealtimeOutgoingVideoSource> RealtimeOutgoingVideoSource::create(Ref<MediaStreamTrackPrivate>&& videoSource)
{
    return RealtimeOutgoingVideoSourceCocoa::create(WTFMove(videoSource));
}

Ref<RealtimeOutgoingVideoSourceCocoa> RealtimeOutgoingVideoSourceCocoa::create(Ref<MediaStreamTrackPrivate>&& videoSource)
{
    return adoptRef(*new RealtimeOutgoingVideoSourceCocoa(WTFMove(videoSource)));
}

RealtimeOutgoingVideoSourceCocoa::RealtimeOutgoingVideoSourceCocoa(Ref<MediaStreamTrackPrivate>&& videoSource)
    : RealtimeOutgoingVideoSource(WTFMove(videoSource))
{
}

RealtimeOutgoingVideoSourceCocoa::~RealtimeOutgoingVideoSourceCocoa() = default;

void RealtimeOutgoingVideoSourceCocoa::videoFrameAvailable(VideoFrame& videoFrame, VideoFrameTimeMetadata)
{
#if !RELEASE_LOG_DISABLED
    if (!(++m_numberOfFrames % 60))
        ALWAYS_LOG(LOGIDENTIFIER, "frame ", m_numberOfFrames);
#endif

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

    auto videoFrameScaling = this->videoFrameScaling();
    bool shouldApplyRotation = m_shouldApplyRotation && m_currentRotation != webrtc::kVideoRotation_0;
    if (!shouldApplyRotation) {
        if (videoFrame.isRemoteProxy()) {
            Ref remoteVideoFrame { videoFrame };
            auto size = videoFrame.presentationSize();
            sendFrame(webrtc::toWebRTCVideoFrameBuffer(&remoteVideoFrame.leakRef(),
                [](auto* pointer) { return static_cast<VideoFrame*>(pointer)->pixelBuffer(); },
                [](auto* pointer) { static_cast<VideoFrame*>(pointer)->deref(); },
                static_cast<int>(size.width() * videoFrameScaling), static_cast<int>(size.height() * videoFrameScaling)));
            return;
        }
        if (auto* webrtcVideoFrame = dynamicDowncast<VideoFrameLibWebRTC>(videoFrame)) {
            auto webrtcBuffer = webrtcVideoFrame->buffer();
            if (videoFrameScaling != 1)
                webrtcBuffer = webrtcBuffer->Scale(webrtcBuffer->width() * videoFrameScaling, webrtcBuffer->height() * videoFrameScaling);
            sendFrame(WTFMove(webrtcBuffer));
            return;
        }
    }

#if ASSERT_ENABLED
    auto pixelFormat = videoFrame.pixelFormat();
    // FIXME: We should use a pixel conformer for other pixel formats and kCVPixelFormatType_32BGRA.
    ASSERT(pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange || pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange || pixelFormat == kCVPixelFormatType_32BGRA);
#endif
    RetainPtr<CVPixelBufferRef> convertedBuffer = videoFrame.pixelBuffer();
    if (shouldApplyRotation)
        convertedBuffer = rotatePixelBuffer(convertedBuffer.get(), m_currentRotation);

    auto webrtcBuffer = webrtc::pixelBufferToFrame(convertedBuffer.get());
    if (videoFrameScaling != 1)
        webrtcBuffer = webrtcBuffer->Scale(webrtcBuffer->width() * videoFrameScaling, webrtcBuffer->height() * videoFrameScaling);

    sendFrame(WTFMove(webrtcBuffer));
}

rtc::scoped_refptr<webrtc::VideoFrameBuffer> RealtimeOutgoingVideoSourceCocoa::createBlackFrame(size_t  width, size_t  height)
{
    return webrtc::pixelBufferToFrame(createBlackPixelBuffer(width, height).get());
}


} // namespace WebCore

#endif // USE(LIBWEBRTC)
