/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#import "RealtimeOutgoingVideoSourceCocoa.h"

#if USE(LIBWEBRTC)

#import "AffineTransform.h"
#import "CVUtilities.h"
#import "ImageRotationSessionVT.h"
#import "Logging.h"
#import "RealtimeVideoUtilities.h"
#import <pal/cf/CoreMediaSoftLink.h>
#import "CoreVideoSoftLink.h"
#import "VideoToolboxSoftLink.h"

namespace WebCore {

static inline unsigned rotationToAngle(webrtc::VideoRotation rotation)
{
    switch (rotation) {
    case webrtc::kVideoRotation_0:
        return 0;
    case webrtc::kVideoRotation_90:
        return 90;
    case webrtc::kVideoRotation_180:
        return 180;
    case webrtc::kVideoRotation_270:
        return 270;
    }
}

RetainPtr<CVPixelBufferRef> RealtimeOutgoingVideoSourceCocoa::rotatePixelBuffer(CVPixelBufferRef pixelBuffer, webrtc::VideoRotation rotation)
{
    ASSERT(rotation);
    if (!rotation)
        return pixelBuffer;

    auto pixelWidth = CVPixelBufferGetWidth(pixelBuffer);
    auto pixelHeight = CVPixelBufferGetHeight(pixelBuffer);
    if (!m_rotationSession || rotation != m_currentRotationSessionAngle || pixelWidth != m_rotatedWidth || pixelHeight != m_rotatedHeight) {
        RELEASE_LOG_INFO(WebRTC, "RealtimeOutgoingVideoSourceCocoa::rotatePixelBuffer creating rotation session for rotation %u", rotationToAngle(rotation));
        AffineTransform transform;
        transform.rotate(rotationToAngle(rotation));
        m_rotationSession = makeUnique<ImageRotationSessionVT>(WTFMove(transform), FloatSize { static_cast<float>(pixelWidth), static_cast<float>(pixelHeight) }, ImageRotationSessionVT::IsCGImageCompatible::No, ImageRotationSessionVT::ShouldUseIOSurface::No);

        m_currentRotationSessionAngle = rotation;
        m_rotatedWidth = pixelWidth;
        m_rotatedHeight = pixelHeight;
    }

    return m_rotationSession->rotate(pixelBuffer);
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)
