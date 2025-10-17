/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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

#include "FloatRect.h"
#include "IntSize.h"
#include <wtf/RetainPtr.h>

typedef struct CGImage *CGImageRef;
typedef struct OpaqueVTPixelTransferSession* VTPixelTransferSessionRef;
typedef struct __CVBuffer *CVPixelBufferRef;
typedef struct __CVPixelBufferPool *CVPixelBufferPoolRef;
typedef struct __IOSurface *IOSurfaceRef;
typedef struct opaqueCMSampleBuffer *CMSampleBufferRef;
OBJC_CLASS NSDictionary;

namespace WTF {
class MediaTime;
}

namespace WebCore {

class FloatRect;
class VideoFrame;
enum class VideoFrameRotation : uint16_t;

class ImageTransferSessionVT {
public:
    static std::unique_ptr<ImageTransferSessionVT> create(uint32_t pixelFormat, bool shouldUseIOSurface = true)
    {
        return std::unique_ptr<ImageTransferSessionVT>(new ImageTransferSessionVT(pixelFormat, shouldUseIOSurface));
    }

    RefPtr<VideoFrame> convertVideoFrame(VideoFrame&, const IntSize&);
    RefPtr<VideoFrame> createVideoFrame(CGImageRef, const WTF::MediaTime&, const IntSize&);
    RefPtr<VideoFrame> createVideoFrame(CMSampleBufferRef, const WTF::MediaTime&, const IntSize&);
    RefPtr<VideoFrame> createVideoFrame(CGImageRef, const WTF::MediaTime&, const IntSize&, VideoFrameRotation, bool mirrored = false);
    RefPtr<VideoFrame> createVideoFrame(CMSampleBufferRef, const WTF::MediaTime&, const IntSize&, VideoFrameRotation, bool mirrored = false);

#if !PLATFORM(MACCATALYST)
    WEBCORE_EXPORT RefPtr<VideoFrame> createVideoFrame(IOSurfaceRef, const WTF::MediaTime&, const IntSize&);
    WEBCORE_EXPORT RefPtr<VideoFrame> createVideoFrame(IOSurfaceRef, const WTF::MediaTime&, const IntSize&, VideoFrameRotation, bool mirrored = false);
#endif

    uint32_t pixelFormat() const { return m_pixelFormat; }
    void setMaximumBufferPoolSize(size_t maxBufferPoolSize) { m_maxBufferPoolSize = maxBufferPoolSize; }

    RetainPtr<CMSampleBufferRef> convertCMSampleBuffer(CMSampleBufferRef, const IntSize&, const WTF::MediaTime* = nullptr);
    void setCroppingRectangle(std::optional<FloatRect>);

private:
    WEBCORE_EXPORT ImageTransferSessionVT(uint32_t pixelFormat, bool shouldUseIOSurface);

#if !PLATFORM(MACCATALYST)
    RetainPtr<CMSampleBufferRef> createCMSampleBuffer(IOSurfaceRef, const WTF::MediaTime&, const IntSize&);
#endif

    RetainPtr<CMSampleBufferRef> createCMSampleBuffer(CVPixelBufferRef, const WTF::MediaTime&, const IntSize&);
    RetainPtr<CMSampleBufferRef> createCMSampleBuffer(CGImageRef, const WTF::MediaTime&, const IntSize&);

    RetainPtr<CVPixelBufferRef> convertPixelBuffer(CVPixelBufferRef, const IntSize&);
    RetainPtr<CVPixelBufferRef> createPixelBuffer(CMSampleBufferRef, const IntSize&);
    RetainPtr<CVPixelBufferRef> createPixelBuffer(CGImageRef, const IntSize&);

    bool setSize(const IntSize&);

    RetainPtr<VTPixelTransferSessionRef> m_transferSession;
    RetainPtr<CVPixelBufferPoolRef> m_outputBufferPool;
    bool m_shouldUseIOSurface { true };
    std::optional<FloatRect> m_croppingRectangle;
    RetainPtr<NSDictionary> m_sourceCroppingDictionary;
    uint32_t m_pixelFormat;
    IntSize m_size;
    size_t m_maxBufferPoolSize { 0 };
};

}
