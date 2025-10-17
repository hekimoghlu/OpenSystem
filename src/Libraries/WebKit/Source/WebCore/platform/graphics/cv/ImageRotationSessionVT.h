/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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

#include "AffineTransform.h"
#include "FloatSize.h"
#include <wtf/TZoneMalloc.h>

typedef struct OpaqueVTImageRotationSession* VTImageRotationSessionRef;
typedef struct __CVBuffer *CVPixelBufferRef;
typedef struct __CVPixelBufferPool* CVPixelBufferPoolRef;

namespace WebCore {

class VideoFrame;

class ImageRotationSessionVT final {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ImageRotationSessionVT, WEBCORE_EXPORT);
public:
    struct RotationProperties {
        bool flipX { false };
        bool flipY { false };
        unsigned angle { 0 };

        friend bool operator==(const RotationProperties&, const RotationProperties&) = default;
        bool isIdentity() const { return !flipX && !flipY && !angle; }
    };

    enum class IsCGImageCompatible : bool { No, Yes };
    enum class ShouldUseIOSurface : bool { No, Yes };

    ImageRotationSessionVT(AffineTransform&&, FloatSize, IsCGImageCompatible, ShouldUseIOSurface = ShouldUseIOSurface::Yes);
    ImageRotationSessionVT(const RotationProperties&, FloatSize, IsCGImageCompatible, ShouldUseIOSurface = ShouldUseIOSurface::Yes);
    ImageRotationSessionVT() = default;

    const std::optional<AffineTransform>& transform() const { return m_transform; }
    const RotationProperties& rotationProperties() const { return m_rotationProperties; }
    const FloatSize& size() { return m_size; }
    const FloatSize& rotatedSize() { return m_rotatedSize; }

    RetainPtr<CVPixelBufferRef> rotate(CVPixelBufferRef);
    WEBCORE_EXPORT RetainPtr<CVPixelBufferRef> rotate(VideoFrame&, const RotationProperties&, IsCGImageCompatible);

private:
    void initialize(const RotationProperties&, FloatSize, IsCGImageCompatible);

    RotationProperties m_rotationProperties;
    FloatSize m_size;
    std::optional<AffineTransform> m_transform;
    OSType m_pixelFormat;
    IsCGImageCompatible m_isCGImageCompatible;
    FloatSize m_rotatedSize;
    RetainPtr<VTImageRotationSessionRef> m_rotationSession;
    RetainPtr<CVPixelBufferPoolRef> m_rotationPool;
    bool m_shouldUseIOSurface { true };
};

}
