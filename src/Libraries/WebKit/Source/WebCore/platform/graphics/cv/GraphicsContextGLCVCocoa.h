/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

#if ENABLE(WEBGL) && ENABLE(VIDEO) && USE(AVFOUNDATION)

#include "GraphicsContextGLCV.h"
#include "ImageOrientation.h"
#include <memory>
#include <wtf/TZoneMalloc.h>

typedef struct __CVBuffer* CVPixelBufferRef;

namespace WebCore {
class GraphicsContextGLCocoa;

// GraphicsContextGLCV implementation for GraphicsContextGLCocoa.
// This class is part of the internal implementation of GraphicsContextGLCocoa.
class GraphicsContextGLCVCocoa final : public GraphicsContextGLCV {
    WTF_MAKE_TZONE_ALLOCATED(GraphicsContextGLCVCocoa);
public:
    static std::unique_ptr<GraphicsContextGLCVCocoa> create(GraphicsContextGLCocoa&);

    ~GraphicsContextGLCVCocoa() final;

    bool copyVideoSampleToTexture(const VideoFrameCV&, PlatformGLObject outputTexture, GCGLint level, GCGLenum internalFormat, GCGLenum format, GCGLenum type, FlipY) final;

    void invalidateKnownTextureContent(GCGLuint texture);
private:
    GraphicsContextGLCVCocoa(GraphicsContextGLCocoa&);

    RetainPtr<CVPixelBufferRef> convertPixelBuffer(CVPixelBufferRef);

    GraphicsContextGLCocoa& m_owner;
    GCGLDisplay m_display { nullptr };
    GCGLContext m_context { nullptr };
    GCGLConfig m_config { nullptr };

    PlatformGLObject m_framebuffer { 0 };
    PlatformGLObject m_yuvVertexBuffer { 0 };
    GCGLint m_yTextureUniformLocation { -1 };
    GCGLint m_uvTextureUniformLocation { -1 };
    GCGLint m_yuvFlipYUniformLocation { -1 };
    GCGLint m_yuvFlipXUniformLocation { -1 };
    GCGLint m_yuvSwapXYUniformLocation { -1 };
    GCGLint m_colorMatrixUniformLocation { -1 };
    GCGLint m_yuvPositionAttributeLocation { -1 };
    GCGLint m_yTextureSizeUniformLocation { -1 };
    GCGLint m_uvTextureSizeUniformLocation { -1 };

    struct TextureContent {
        intptr_t surface { 0 };
        uint32_t surfaceID { 0 };
        uint32_t surfaceSeed { 0 };
        GCGLint level { 0 };
        GCGLenum internalFormat { 0 };
        GCGLenum format { 0 };
        GCGLenum type { 0 };
        FlipY unpackFlipY { FlipY::No };
        ImageOrientation orientation;

        friend bool operator==(const TextureContent&, const TextureContent&) = default;
    };
    using TextureContentMap = UncheckedKeyHashMap<GCGLuint, TextureContent, IntHash<GCGLuint>, WTF::UnsignedWithZeroKeyHashTraits<GCGLuint>>;
    TextureContentMap m_knownContent;
};

}

#endif
