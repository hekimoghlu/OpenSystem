/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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

#if USE(LIBWEBRTC)

#include "RealtimeOutgoingVideoSource.h"
#include <webrtc/api/video/video_rotation.h>

using CVPixelBufferPoolRef = struct __CVPixelBufferPool*;
using CVPixelBufferRef = struct __CVBuffer*;

namespace WebCore {

class ImageRotationSessionVT;

class RealtimeOutgoingVideoSourceCocoa final : public RealtimeOutgoingVideoSource {
public:
    static Ref<RealtimeOutgoingVideoSourceCocoa> create(Ref<MediaStreamTrackPrivate>&&);

    virtual ~RealtimeOutgoingVideoSourceCocoa();

private:
    explicit RealtimeOutgoingVideoSourceCocoa(Ref<MediaStreamTrackPrivate>&&);

    rtc::scoped_refptr<webrtc::VideoFrameBuffer> createBlackFrame(size_t width, size_t height) final;

    // RealtimeMediaSource::VideoFrameObserver API
    void videoFrameAvailable(VideoFrame&, VideoFrameTimeMetadata) final;

    RetainPtr<CVPixelBufferRef> rotatePixelBuffer(CVPixelBufferRef, webrtc::VideoRotation);
    CVPixelBufferPoolRef pixelBufferPool(size_t width, size_t height);

    std::unique_ptr<ImageRotationSessionVT> m_rotationSession;
    webrtc::VideoRotation m_currentRotationSessionAngle { webrtc::kVideoRotation_0 };
    size_t m_rotatedWidth { 0 };
    size_t m_rotatedHeight { 0 };
    OSType m_rotatedFormat;

#if !RELEASE_LOG_DISABLED
    size_t m_numberOfFrames { 0 };
#endif
};

} // namespace WebCore

#endif // USE(LIBWEBRTC)
