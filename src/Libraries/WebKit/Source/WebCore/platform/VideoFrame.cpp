/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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
#include "VideoFrame.h"

#if ENABLE(VIDEO)

#if USE(GSTREAMER)
#include "VideoFrameGStreamer.h"
#endif

#if !PLATFORM(COCOA) && !USE(GSTREAMER)
#include "ImageOrientation.h"
#endif

namespace WebCore {

VideoFrame::VideoFrame(MediaTime presentationTime, bool isMirrored, Rotation rotation, PlatformVideoColorSpace&& colorSpace)
    : m_presentationTime(presentationTime)
    , m_isMirrored(isMirrored)
    , m_rotation(rotation)
    , m_colorSpace(WTFMove(colorSpace))
{
}

void VideoFrame::initializeCharacteristics(MediaTime presentationTime, bool isMirrored, Rotation rotation)
{
    const_cast<MediaTime&>(m_presentationTime) = presentationTime;
    const_cast<bool&>(m_isMirrored) = isMirrored;
    const_cast<Rotation&>(m_rotation) = rotation;
}

Ref<VideoFrame> VideoFrame::updateTimestamp(MediaTime mediaTime, ShouldCloneWithDifferentTimestamp shouldCloneWithDifferentTimestamp)
{
    if (m_presentationTime == mediaTime)
        return *this;

    Ref updatedVideoFrame = shouldCloneWithDifferentTimestamp == ShouldCloneWithDifferentTimestamp::Yes ? clone() : Ref { *this };
    const_cast<MediaTime&>(updatedVideoFrame->m_presentationTime) = mediaTime;
    return updatedVideoFrame;
}

#if !PLATFORM(COCOA) && !USE(GSTREAMER)
RefPtr<VideoFrame> VideoFrame::fromNativeImage(NativeImage&)
{
    // FIXME: Add support.
    return nullptr;
}

RefPtr<VideoFrame> createFromPixelBuffer(Ref<PixelBuffer>&&, PlatformVideoColorSpace&&)
{
    // FIXME: Add support.
    return nullptr;
}

RefPtr<VideoFrame> VideoFrame::createNV12(std::span<const uint8_t>, size_t, size_t, const ComputedPlaneLayout&, const ComputedPlaneLayout&, PlatformVideoColorSpace&&)
{
    // FIXME: Add support.
    return nullptr;
}

RefPtr<VideoFrame> VideoFrame::createRGBA(std::span<const uint8_t>, size_t, size_t, const ComputedPlaneLayout&, PlatformVideoColorSpace&&)
{
    // FIXME: Add support.
    return nullptr;
}

RefPtr<VideoFrame> VideoFrame::createBGRA(std::span<const uint8_t>, size_t, size_t, const ComputedPlaneLayout&, PlatformVideoColorSpace&&)
{
    // FIXME: Add support.
    return nullptr;
}

RefPtr<VideoFrame> VideoFrame::createI420(std::span<const uint8_t>, size_t, size_t, const ComputedPlaneLayout&, const ComputedPlaneLayout&, const ComputedPlaneLayout&, PlatformVideoColorSpace&&)
{
    // FIXME: Add support.
    return nullptr;
}

RefPtr<VideoFrame> VideoFrame::createI420A(std::span<const uint8_t>, size_t, size_t, const ComputedPlaneLayout&, const ComputedPlaneLayout&, const ComputedPlaneLayout&, const ComputedPlaneLayout&, PlatformVideoColorSpace&&)
{
    // FIXME: Add support.
    return nullptr;
}

void VideoFrame::copyTo(std::span<uint8_t>, VideoPixelFormat, Vector<ComputedPlaneLayout>&&, CopyCallback&& callback)
{
    // FIXME: Add support.
    callback({ });
}

void VideoFrame::draw(GraphicsContext&, const FloatRect&, ImageOrientation, bool)
{
    // FIXME: Add support.
}
#endif // !PLATFORM(COCOA)

}

#endif // ENABLE(VIDEO)
