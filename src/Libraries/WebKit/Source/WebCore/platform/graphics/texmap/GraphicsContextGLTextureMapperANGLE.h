/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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

#if ENABLE(WEBGL) && USE(TEXTURE_MAPPER)

#include "GLContextWrapper.h"
#include "GraphicsContextGLANGLE.h"

namespace WebCore {

class TextureMapperGCGLPlatformLayer;

class GraphicsContextGLTextureMapperANGLE : public GLContextWrapper, public GraphicsContextGLANGLE {
public:
    WEBCORE_EXPORT static RefPtr<GraphicsContextGLTextureMapperANGLE> create(WebCore::GraphicsContextGLAttributes&&);
    virtual ~GraphicsContextGLTextureMapperANGLE();

    // GraphicsContextGLANGLE overrides.
    WEBCORE_EXPORT RefPtr<GraphicsLayerContentsDisplayDelegate> layerContentsDisplayDelegate() final;
#if ENABLE(VIDEO)
    bool copyTextureFromVideoFrame(VideoFrame&, PlatformGLObject texture, GCGLenum target, GCGLint level, GCGLenum internalFormat, GCGLenum format, GCGLenum type, bool premultiplyAlpha, bool flipY) final;
#endif
#if ENABLE(MEDIA_STREAM) || ENABLE(WEB_CODECS)
    RefPtr<VideoFrame> surfaceBufferToVideoFrame(SurfaceBuffer) final;
#endif
    RefPtr<PixelBuffer> readCompositedResults() final;

    bool reshapeDrawingBuffer() override;
    void prepareForDisplay() override;
#if ENABLE(WEBXR)
    bool addFoveation(IntSize, IntSize, IntSize, std::span<const GCGLfloat>, std::span<const GCGLfloat>, std::span<const GCGLfloat>) final;
    void enableFoveation(GCGLuint) final;
    void disableFoveation() final;
#endif

protected:
    explicit GraphicsContextGLTextureMapperANGLE(WebCore::GraphicsContextGLAttributes&&);

    RefPtr<GraphicsLayerContentsDisplayDelegate> m_layerContentsDisplayDelegate;

private:
    bool platformInitializeContext() final;
    bool platformInitialize() override;

    void swapCompositorTexture();

#if USE(COORDINATED_GRAPHICS) && USE(LIBEPOXY)
    GCGLuint setupCurrentTexture();
#endif

    // GLContextWrapper
    GLContextWrapper::Type type() const override;
    bool makeCurrentImpl() override;
    bool unmakeCurrentImpl() override;

    GCGLuint m_compositorTexture { 0 };
    bool m_isCompositorTextureInitialized { false };

#if USE(COORDINATED_GRAPHICS) && USE(LIBEPOXY)
    GCGLuint m_textureID { 0 };
    GCGLuint m_compositorTextureID { 0 };
#endif

#if !USE(COORDINATED_GRAPHICS)
    std::unique_ptr<TextureMapperGCGLPlatformLayer> m_texmapLayer;

    friend class TextureMapperGCGLPlatformLayer;
#endif
};

} // namespace WebCore

#endif // ENABLE(WEBGL) && USE(TEXTURE_MAPPER)
