/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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
#include "TextureMapperGCGLPlatformLayer.h"

#if ENABLE(WEBGL) && USE(TEXTURE_MAPPER) && !USE(COORDINATED_GRAPHICS)

#include "ANGLEHeaders.h"
#include "BitmapTexture.h"
#include "GLContext.h"
#include "TextureMapper.h"
#include "TextureMapperFlags.h"
#include "TextureMapperGLHeaders.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextureMapperGCGLPlatformLayer);

TextureMapperGCGLPlatformLayer::TextureMapperGCGLPlatformLayer(GraphicsContextGLTextureMapperANGLE& context)
    : m_context(context)
{
}

TextureMapperGCGLPlatformLayer::~TextureMapperGCGLPlatformLayer()
{
    if (client())
        client()->platformLayerWillBeDestroyed();
}

void TextureMapperGCGLPlatformLayer::paintToTextureMapper(TextureMapper& textureMapper, const FloatRect& targetRect, const TransformationMatrix& matrix, float opacity)
{
    if (!m_context.m_isCompositorTextureInitialized)
        return;

    auto attrs = m_context.contextAttributes();
    OptionSet<TextureMapperFlags> flags = TextureMapperFlags::ShouldFlipTexture;
    if (attrs.alpha)
        flags.add(TextureMapperFlags::ShouldBlend);
    textureMapper.drawTexture(m_context.m_compositorTexture, flags, targetRect, matrix, opacity);
}

} // namespace WebCore

#endif // ENABLE(WEBGL) && USE(TEXTURE_MAPPER) && !USE(COORDINATED_GRAPHICS)
