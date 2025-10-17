/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#include "CoordinatedBackingStore.h"

#if USE(COORDINATED_GRAPHICS)
#include "BitmapTexture.h"
#include "CoordinatedTileBuffer.h"
#include "GraphicsLayer.h"
#include "TextureMapper.h"
#include <wtf/SystemTracing.h>

namespace WebCore {

void CoordinatedBackingStore::createTile(uint32_t id)
{
    m_tiles.add(id, CoordinatedBackingStoreTile(m_scale));
}

void CoordinatedBackingStore::removeTile(uint32_t id)
{
    ASSERT(m_tiles.contains(id));
    m_tiles.remove(id);
}

void CoordinatedBackingStore::updateTile(uint32_t id, const IntRect& sourceRect, const IntRect& tileRect, RefPtr<CoordinatedTileBuffer>&& buffer, const IntPoint& offset)
{
    auto it = m_tiles.find(id);
    ASSERT(it != m_tiles.end());
    it->value.addUpdate({ WTFMove(buffer), sourceRect, tileRect, offset });
}

void CoordinatedBackingStore::processPendingUpdates(TextureMapper& textureMapper)
{
    for (auto& tile : m_tiles.values())
        tile.processPendingUpdates(textureMapper);
}

void CoordinatedBackingStore::resize(const FloatSize& size, float scale)
{
    m_size = size;
    m_scale = scale;
}

void CoordinatedBackingStore::paintToTextureMapper(TextureMapper& textureMapper, const FloatRect& targetRect, const TransformationMatrix& transform, float opacity)
{
    if (m_tiles.isEmpty())
        return;

    ASSERT(!m_size.isZero());
    Vector<CoordinatedBackingStoreTile*, 16> tilesToPaint;
    Vector<CoordinatedBackingStoreTile*, 16> previousTilesToPaint;

    // We have to do this every time we paint, in case the opacity has changed.
    FloatRect coveredRect;
    for (auto& tile : m_tiles.values()) {
        if (!tile.canBePainted())
            continue;

        if (tile.scale() == m_scale) {
            tilesToPaint.append(&tile);
            coveredRect.unite(tile.rect());
            continue;
        }

        // Only show the previous tile if the opacity is high, otherwise effect looks like a bug.
        // We show the previous-scale tile anyway if it doesn't intersect with any current-scale tile.
        if (opacity < 0.95 && coveredRect.intersects(tile.rect()))
            continue;

        previousTilesToPaint.append(&tile);
    }

    // targetRect is on the contents coordinate system, so we must compare two rects on the contents coordinate system.
    // See CoodinatedBackingStoreProxy.
    FloatRect layerRect = { { }, m_size };
    TransformationMatrix adjustedTransform = transform * TransformationMatrix::rectToRect(layerRect, targetRect);

    auto paintTile = [&](CoordinatedBackingStoreTile& tile) {
        textureMapper.drawTexture(tile.texture(), tile.rect(), adjustedTransform, opacity, allTileEdgesExposed(layerRect, tile.rect()) ? TextureMapper::AllEdgesExposed::Yes : TextureMapper::AllEdgesExposed::No);
    };

    for (auto* tile : previousTilesToPaint)
        paintTile(*tile);
    for (auto* tile : tilesToPaint)
        paintTile(*tile);
}

void CoordinatedBackingStore::drawBorder(TextureMapper& textureMapper, const Color& borderColor, float borderWidth, const FloatRect& targetRect, const TransformationMatrix& transform)
{
    FloatRect layerRect = { { }, m_size };
    TransformationMatrix adjustedTransform = transform * TransformationMatrix::rectToRect(layerRect, targetRect);
    for (auto& tile : m_tiles.values())
        textureMapper.drawBorder(borderColor, borderWidth, tile.rect(), adjustedTransform);
}

void CoordinatedBackingStore::drawRepaintCounter(TextureMapper& textureMapper, int repaintCount, const Color& borderColor, const FloatRect& targetRect, const TransformationMatrix& transform)
{
    FloatRect layerRect = { { }, m_size };
    TransformationMatrix adjustedTransform = transform * TransformationMatrix::rectToRect(layerRect, targetRect);
    for (auto& tile : m_tiles.values())
        textureMapper.drawNumber(repaintCount, borderColor, tile.rect().location(), adjustedTransform);
}

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
