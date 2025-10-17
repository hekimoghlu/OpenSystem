/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#include "TextureMapperSparseBackingStore.h"
#include <wtf/TZoneMallocInlines.h>

#if USE(GRAPHICS_LAYER_WC)
#include "IntRect.h"
#include "TextureMapper.h"

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextureMapperSparseBackingStore);

void TextureMapperSparseBackingStore::setSize(const IntSize& size)
{
    if (m_size == size)
        return;
    m_size = size;
    m_tiles.clear();
}

TransformationMatrix TextureMapperSparseBackingStore::adjustedTransformForRect(const FloatRect& targetRect)
{
    return TransformationMatrix::rectToRect({ { }, m_size }, targetRect);
}

void TextureMapperSparseBackingStore::paintToTextureMapper(TextureMapper& textureMapper, const FloatRect& targetRect, const TransformationMatrix& transform, float opacity)
{
    IntRect rect = { { }, m_size };
    TransformationMatrix adjustedTransform = transform * adjustedTransformForRect(targetRect);
    for (auto& iterator : m_tiles)
        iterator.value->paint(textureMapper, adjustedTransform, opacity, allTileEdgesExposed(rect, iterator.value->rect()));
}

void TextureMapperSparseBackingStore::drawBorder(TextureMapper& textureMapper, const Color& borderColor, float borderWidth, const FloatRect& targetRect, const TransformationMatrix& transform)
{
    TransformationMatrix adjustedTransform = transform * adjustedTransformForRect(targetRect);
    for (auto& iterator : m_tiles)
        textureMapper.drawBorder(borderColor, borderWidth, iterator.value->rect(), adjustedTransform);
}

void TextureMapperSparseBackingStore::drawRepaintCounter(TextureMapper& textureMapper, int repaintCount, const Color& borderColor, const FloatRect& targetRect, const TransformationMatrix& transform)
{
    TransformationMatrix adjustedTransform = transform * adjustedTransformForRect(targetRect);
    for (auto& iterator : m_tiles)
        textureMapper.drawNumber(repaintCount, borderColor, iterator.value->rect().location(), adjustedTransform);
}

void TextureMapperSparseBackingStore::updateContents(const TileIndex& index, Image& image, const IntRect& dirtyRect)
{
    auto addResult = m_tiles.ensure(index, [&]() {
        return makeUnique<TextureMapperTile>(dirtyRect);
    });
    addResult.iterator->value->updateContents(&image, dirtyRect);
}

void TextureMapperSparseBackingStore::removeTile(const TileIndex& index)
{
    m_tiles.remove(index);
}

} // namespace WebCore

#endif // USE(GRAPHICS_LAYER_WC)
