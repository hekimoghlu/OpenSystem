/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "TextureMapperTile.h"

#include "BitmapTexture.h"
#include "Image.h"
#include "TextureMapper.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextureMapperTile);

class GraphicsLayer;

TextureMapperTile::TextureMapperTile(const FloatRect& rect)
    : m_rect(rect)
{
}

TextureMapperTile::~TextureMapperTile() = default;

RefPtr<BitmapTexture> TextureMapperTile::texture() const
{
    return m_texture;
}

void TextureMapperTile::setTexture(BitmapTexture* texture)
{
    m_texture = texture;
}

void TextureMapperTile::updateContents(Image* image, const IntRect& dirtyRect)
{
    IntRect targetRect = enclosingIntRect(m_rect);
    targetRect.intersect(dirtyRect);
    if (targetRect.isEmpty())
        return;
    IntPoint sourceOffset = targetRect.location();

    // Normalize sourceRect to the buffer's coordinates.
    sourceOffset.move(-dirtyRect.x(), -dirtyRect.y());

    // Normalize targetRect to the texture's coordinates.
    targetRect.move(-m_rect.x(), -m_rect.y());
    if (!m_texture) {
        OptionSet<BitmapTexture::Flags> flags;
        if (!image->currentFrameKnownToBeOpaque())
            flags.add(BitmapTexture::Flags::SupportsAlpha);
        m_texture = BitmapTexture::create(targetRect.size(), flags);
    }

    auto nativeImage = image->currentNativeImage();
    m_texture->updateContents(nativeImage.get(), targetRect, sourceOffset);
}

void TextureMapperTile::updateContents(GraphicsLayer* sourceLayer, const IntRect& dirtyRect, float scale)
{
    IntRect targetRect = enclosingIntRect(m_rect);
    targetRect.intersect(dirtyRect);
    if (targetRect.isEmpty())
        return;
    IntPoint sourceOffset = targetRect.location();

    // Normalize targetRect to the texture's coordinates.
    targetRect.move(-m_rect.x(), -m_rect.y());

    if (!m_texture)
        m_texture = BitmapTexture::create(targetRect.size(), { BitmapTexture::Flags::SupportsAlpha });

    m_texture->updateContents(sourceLayer, targetRect, sourceOffset, scale);
}

void TextureMapperTile::paint(TextureMapper& textureMapper, const TransformationMatrix& transform, float opacity, bool allEdgesExposed)
{
    if (texture().get())
        textureMapper.drawTexture(*texture().get(), rect(), transform, opacity, allEdgesExposed ? TextureMapper::AllEdgesExposed::Yes : TextureMapper::AllEdgesExposed::No);
}

} // namespace WebCore
