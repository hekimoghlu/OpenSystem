/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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

#include "ClipStack.h"
#include "FilterOperation.h"
#include "IntPoint.h"
#include "IntRect.h"
#include "IntSize.h"
#include "PixelFormat.h"
#include "TextureMapperGLHeaders.h"
#include <wtf/OptionSet.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class GraphicsLayer;
class NativeImage;
class TextureMapper;
enum class TextureMapperFlags : uint16_t;

class BitmapTexture final : public ThreadSafeRefCounted<BitmapTexture> {
public:
    enum class Flags : uint8_t {
        SupportsAlpha = 1 << 0,
        DepthBuffer = 1 << 1,
    };

    static Ref<BitmapTexture> create(const IntSize& size, OptionSet<Flags> flags = { })
    {
        return adoptRef(*new BitmapTexture(size, flags));
    }

    WEBCORE_EXPORT ~BitmapTexture();

    const IntSize& size() const { return m_size; };
    OptionSet<Flags> flags() const { return m_flags; }
    bool isOpaque() const { return !m_flags.contains(Flags::SupportsAlpha); }

    void bindAsSurface();
    void initializeStencil();
    void initializeDepthBuffer();
    uint32_t id() const { return m_id; }

    void updateContents(NativeImage*, const IntRect&, const IntPoint& offset);
    void updateContents(GraphicsLayer*, const IntRect& target, const IntPoint& offset, float scale = 1);
    void updateContents(const void* srcData, const IntRect& targetRect, const IntPoint& sourceOffset, int bytesPerLine, PixelFormat);

    void swapTexture(BitmapTexture&);
    void reset(const IntSize&, OptionSet<Flags> = { });

    int numberOfBytes() const { return size().width() * size().height() * 32 >> 3; }

    RefPtr<const FilterOperation> filterOperation() const { return m_filterOperation; }
    void setFilterOperation(RefPtr<const FilterOperation>&& filterOperation) { m_filterOperation = WTFMove(filterOperation); }

    ClipStack& clipStack() { return m_clipStack; }

    void copyFromExternalTexture(GLuint textureID);
    void copyFromExternalTexture(BitmapTexture& sourceTexture, const IntRect& sourceRect, const IntSize& destinationOffset);
    void copyFromExternalTexture(GLuint sourceTextureID, const IntRect& targetRect, const IntSize& sourceOffset);

    OptionSet<TextureMapperFlags> colorConvertFlags() const;

private:
    BitmapTexture(const IntSize&, OptionSet<Flags>);

    void clearIfNeeded();
    void createFboIfNeeded();

    OptionSet<Flags> m_flags;
    IntSize m_size;
    GLuint m_id { 0 };
    GLuint m_fbo { 0 };
    GLuint m_depthBufferObject { 0 };
    GLuint m_stencilBufferObject { 0 };
    bool m_stencilBound { false };
    bool m_shouldClear { true };
    ClipStack m_clipStack;
    RefPtr<const FilterOperation> m_filterOperation;
    PixelFormat m_pixelFormat { PixelFormat::RGBA8 };
};

} // namespace WebCore
