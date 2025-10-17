/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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

#include "RealtimeIncomingVideoSource.h"

using CVPixelBufferPoolRef = struct __CVPixelBufferPool*;
using CVPixelBufferRef = struct __CVBuffer*;

namespace webrtc {
enum class BufferType;
};

namespace WebCore {

enum class VideoFrameRotation : uint16_t;

class RealtimeIncomingVideoSourceCocoa final : public RealtimeIncomingVideoSource {
public:
    static Ref<RealtimeIncomingVideoSourceCocoa> create(rtc::scoped_refptr<webrtc::VideoTrackInterface>&&, String&&);

private:
    RealtimeIncomingVideoSourceCocoa(rtc::scoped_refptr<webrtc::VideoTrackInterface>&&, String&&);
    RetainPtr<CVPixelBufferRef> pixelBufferFromVideoFrame(const webrtc::VideoFrame&);
    CVPixelBufferPoolRef pixelBufferPool(size_t width, size_t height, webrtc::BufferType);
    RefPtr<VideoFrame> toVideoFrame(const webrtc::VideoFrame&, VideoFrameRotation);
    Ref<VideoFrame> createVideoSampleFromCVPixelBuffer(RetainPtr<CVPixelBufferRef>&&, VideoFrameRotation, int64_t);

    // rtc::VideoSinkInterface
    void OnFrame(const webrtc::VideoFrame&) final;

    RetainPtr<CVPixelBufferRef> m_blackFrame;
    int m_blackFrameWidth { 0 };
    int m_blackFrameHeight { 0 };
#if !RELEASE_LOG_DISABLED
    size_t m_numberOfFrames { 0 };
#endif
    Lock m_pixelBufferPoolLock;
    RetainPtr<CVPixelBufferPoolRef> m_pixelBufferPool WTF_GUARDED_BY_LOCK(m_pixelBufferPoolLock);
    size_t m_pixelBufferPoolWidth { 0 };
    size_t m_pixelBufferPoolHeight { 0 };
    webrtc::BufferType m_pixelBufferPoolBufferType;
};

} // namespace WebCore

#endif // USE(LIBWEBRTC)
