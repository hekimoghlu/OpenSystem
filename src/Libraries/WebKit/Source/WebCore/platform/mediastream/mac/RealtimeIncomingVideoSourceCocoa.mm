/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
#import "config.h"
#import "RealtimeIncomingVideoSourceCocoa.h"

#if USE(LIBWEBRTC)

#import "CVUtilities.h"
#import "Logging.h"
#import "VideoFrameCV.h"
#import "VideoFrameLibWebRTC.h"
#import <wtf/cf/TypeCastsCF.h>

ALLOW_UNUSED_PARAMETERS_BEGIN
#import <webrtc/webkit_sdk/WebKit/WebKitUtilities.h>
ALLOW_UNUSED_PARAMETERS_END

#import "CoreVideoSoftLink.h"
#import <pal/cf/CoreMediaSoftLink.h>

namespace WebCore {

Ref<RealtimeIncomingVideoSource> RealtimeIncomingVideoSource::create(rtc::scoped_refptr<webrtc::VideoTrackInterface>&& videoTrack, String&& trackId)
{
    auto source = RealtimeIncomingVideoSourceCocoa::create(WTFMove(videoTrack), WTFMove(trackId));
    source->start();
    return WTFMove(source);
}

Ref<RealtimeIncomingVideoSourceCocoa> RealtimeIncomingVideoSourceCocoa::create(rtc::scoped_refptr<webrtc::VideoTrackInterface>&& videoTrack, String&& trackId)
{
    return adoptRef(*new RealtimeIncomingVideoSourceCocoa(WTFMove(videoTrack), WTFMove(trackId)));
}

RealtimeIncomingVideoSourceCocoa::RealtimeIncomingVideoSourceCocoa(rtc::scoped_refptr<webrtc::VideoTrackInterface>&& videoTrack, String&& videoTrackId)
    : RealtimeIncomingVideoSource(WTFMove(videoTrack), WTFMove(videoTrackId))
{
}

CVPixelBufferPoolRef RealtimeIncomingVideoSourceCocoa::pixelBufferPool(size_t width, size_t height, webrtc::BufferType bufferType) WTF_IGNORES_THREAD_SAFETY_ANALYSIS
{
    if (!m_pixelBufferPool || m_pixelBufferPoolWidth != width || m_pixelBufferPoolHeight != height || m_pixelBufferPoolBufferType != bufferType) {
        OSType poolBufferType;
        switch (bufferType) {
        case webrtc::BufferType::I420:
            poolBufferType = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
            break;
        case webrtc::BufferType::I010:
            poolBufferType = kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange;
            break;
        default:
            return nullptr;
        }

        auto result = createInMemoryCVPixelBufferPool(width, height, poolBufferType);
        if (!result) {
            RELEASE_LOG_ERROR(WebRTC, "RealtimeIncomingVideoSourceCocoa failed creating buffer pool with error %d", result.error());
            return nullptr;
        }

        m_pixelBufferPool = WTFMove(*result);
        m_pixelBufferPoolWidth = width;
        m_pixelBufferPoolHeight = height;
        m_pixelBufferPoolBufferType = bufferType;
    }
    return m_pixelBufferPool.get();
}

Ref<VideoFrame> RealtimeIncomingVideoSourceCocoa::createVideoSampleFromCVPixelBuffer(RetainPtr<CVPixelBufferRef>&& pixelBuffer, VideoFrame::Rotation rotation, int64_t timeStamp)
{
    return VideoFrameCV::create(MediaTime(timeStamp, 1000000), false, rotation, WTFMove(pixelBuffer));
}

RefPtr<VideoFrame> RealtimeIncomingVideoSourceCocoa::toVideoFrame(const webrtc::VideoFrame& frame, VideoFrame::Rotation rotation)
{
    if (muted()) {
        if (!m_blackFrame || m_blackFrameWidth != frame.width() || m_blackFrameHeight != frame.height()) {
            m_blackFrameWidth = frame.width();
            m_blackFrameHeight = frame.height();
            m_blackFrame = createBlackPixelBuffer(m_blackFrameWidth, m_blackFrameHeight);
        }
        return createVideoSampleFromCVPixelBuffer(m_blackFrame.get(), rotation, frame.timestamp_us());
    }

    if (auto* provider = videoFrameBufferProvider(frame)) {
        // The only supported provider is VideoFrame.
        auto* videoFrame = static_cast<VideoFrame*>(provider);
        videoFrame->initializeCharacteristics(MediaTime { frame.timestamp_us(), 1000000 }, false, rotation);
        return videoFrame;
    }

    // If we already have a CVPixelBufferRef, use it directly.
    if (auto pixelBuffer = adoptCF(webrtc::copyPixelBufferForFrame(frame)))
        return createVideoSampleFromCVPixelBuffer(WTFMove(pixelBuffer), rotation, frame.timestamp_us());

    // In case of in memory libwebrtc samples, we have non interleaved YUV data, let's lazily create CVPixelBuffers if needed.
    return VideoFrameLibWebRTC::create(MediaTime(frame.timestamp_us(), 1000000), false, rotation, VideoFrameLibWebRTC::colorSpaceFromFrame(frame), frame.video_frame_buffer(), [protectedThis = Ref { *this }, this](auto& buffer) {
        return adoptCF(webrtc::createPixelBufferFromFrameBuffer(buffer, [this](size_t width, size_t height, webrtc::BufferType bufferType) -> CVPixelBufferRef {
            Locker lock(m_pixelBufferPoolLock);
            auto pixelBufferPool = this->pixelBufferPool(width, height, bufferType);
            if (!pixelBufferPool)
                return nullptr;
            CVPixelBufferRef pixelBuffer = nullptr;
            auto status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, m_pixelBufferPool.get(), &pixelBuffer);
            if (status != kCVReturnSuccess) {
                ERROR_LOG_IF(loggerPtr(), LOGIDENTIFIER, "Failed creating a pixel buffer with error ", status);
                return nullptr;
            }
            return pixelBuffer;
        }));
    });
}

void RealtimeIncomingVideoSourceCocoa::OnFrame(const webrtc::VideoFrame& webrtcVideoFrame)
{
    if (!isProducingData())
        return;

    unsigned width = webrtcVideoFrame.width();
    unsigned height = webrtcVideoFrame.height();

    VideoFrame::Rotation rotation;
    switch (webrtcVideoFrame.rotation()) {
    case webrtc::kVideoRotation_0:
        rotation = VideoFrame::Rotation::None;
        break;
    case webrtc::kVideoRotation_180:
        rotation = VideoFrame::Rotation::UpsideDown;
        break;
    case webrtc::kVideoRotation_90:
        rotation = VideoFrame::Rotation::Right;
        std::swap(width, height);
        break;
    case webrtc::kVideoRotation_270:
        rotation = VideoFrame::Rotation::Left;
        std::swap(width, height);
        break;
    }

#if !RELEASE_LOG_DISABLED
    ALWAYS_LOG_IF(loggerPtr() && !(++m_numberOfFrames % 60), LOGIDENTIFIER, "frame ", m_numberOfFrames, ", rotation ", webrtcVideoFrame.rotation(), " size ", width, "x", height);
#endif

    auto videoFrame = toVideoFrame(webrtcVideoFrame, rotation);
    if (!videoFrame)
        return;

    notifyNewFrame();

    setIntrinsicSize(IntSize(width, height));
    videoFrameAvailable(*videoFrame, metadataFromVideoFrame(webrtcVideoFrame));
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)
