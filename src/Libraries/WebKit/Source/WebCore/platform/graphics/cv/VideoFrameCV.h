/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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

#if ENABLE(VIDEO) && USE(AVFOUNDATION)

#include "ImageOrientation.h"
#include "VideoFrame.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/RetainPtr.h>

using CMSampleBufferRef = struct opaqueCMSampleBuffer*;

namespace WebCore {

class PixelBuffer;

class VideoFrameCV : public VideoFrame {
public:
    WEBCORE_EXPORT static Ref<VideoFrameCV> create(MediaTime presentationTime, bool isMirrored, Rotation, RetainPtr<CVPixelBufferRef>&&, std::optional<PlatformVideoColorSpace>&& = { });
    WEBCORE_EXPORT static Ref<VideoFrameCV> create(CMSampleBufferRef, bool isMirrored, Rotation);
    WEBCORE_EXPORT ~VideoFrameCV();

    CVPixelBufferRef pixelBuffer() const final { return m_pixelBuffer.get(); }
    ImageOrientation orientation() const;

    // VideoFrame overrides.
    WEBCORE_EXPORT WebCore::IntSize presentationSize() const final;
    WEBCORE_EXPORT uint32_t pixelFormat() const final;
    WEBCORE_EXPORT void setOwnershipIdentity(const ProcessIdentity&) final;
    bool isCV() const final { return true; }

private:
    friend struct IPC::ArgumentCoder<VideoFrameCV, void>;
    WEBCORE_EXPORT VideoFrameCV(MediaTime presentationTime, bool isMirrored, Rotation, RetainPtr<CVPixelBufferRef>&&, std::optional<PlatformVideoColorSpace>&&);
    VideoFrameCV(MediaTime presentationTime, bool isMirrored, Rotation, RetainPtr<CVPixelBufferRef>&&, PlatformVideoColorSpace&&);

    Ref<VideoFrame> clone() final;

    const RetainPtr<CVPixelBufferRef> m_pixelBuffer;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::VideoFrameCV)
    static bool isType(const WebCore::VideoFrame& videoFrame) { return videoFrame.isCV(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif
