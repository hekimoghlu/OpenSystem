/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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
#ifndef LegacyTileGrid_h
#define LegacyTileGrid_h

#if PLATFORM(IOS_FAMILY)

#include "IntPoint.h"
#include "IntPointHash.h"
#include "IntRect.h"
#include "IntSize.h"
#include "LegacyTileCache.h"
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

#define LOG_TILING 0

@class CALayer;

namespace WebCore {

class LegacyTileGridTile;

class LegacyTileGrid {
    WTF_MAKE_TZONE_ALLOCATED(LegacyTileGrid);
public:
    typedef IntPoint TileIndex;

    LegacyTileGrid(LegacyTileCache&, const IntSize&);
    ~LegacyTileGrid();

    LegacyTileCache& tileCache() const { return m_tileCache.get(); }
    Ref<LegacyTileCache> protectedTileCache() const { return tileCache(); }

    CALayer *tileHostLayer() const;
    IntRect bounds() const;
    unsigned tileCount() const;

    float scale() const { return m_scale; }
    void setScale(float scale) { m_scale = scale; }

    IntRect visibleRect() const;

    void createTiles(LegacyTileCache::SynchronousTileCreationMode);

    void dropAllTiles();
    void dropInvalidTiles();
    void dropTilesOutsideRect(const IntRect&);
    void dropTilesIntersectingRect(const IntRect&);
    // Drops tiles that intersect dropRect but do not intersect keepRect.
    void dropTilesBetweenRects(const IntRect& dropRect, const IntRect& keepRect);
    bool dropDistantTiles(unsigned tilesNeeded, double shortestDistance);

    void addTilesCoveringRect(const IntRect&);

    bool tilesCover(const IntRect&) const;
    void centerTileGridOrigin(const IntRect& visibleRect);
    void invalidateTiles(const IntRect& dirtyRect);

    void updateTileOpacity();
    void updateTileBorderVisibility();
    void updateHostLayerSize();
    bool checkDoSingleTileLayout();

    bool hasTiles() const { return !m_tiles.isEmpty(); }

    IntRect calculateCoverRect(const IntRect& visibleRect, bool& centerGrid);

    // Logging
    void dumpTiles();

private:
    double tileDistance2(const IntRect& visibleRect, const IntRect& tileRect) const;
    unsigned tileByteSize() const;

    void addTileForIndex(const TileIndex&);

    RefPtr<LegacyTileGridTile> tileForIndex(const TileIndex&) const;
    IntRect tileRectForIndex(const TileIndex&) const;
    RefPtr<LegacyTileGridTile> tileForPoint(const IntPoint&) const;
    TileIndex tileIndexForPoint(const IntPoint&) const;

    IntRect adjustCoverRectForPageBounds(const IntRect&) const;
    bool shouldUseMinimalTileCoverage() const;

private:        
    WeakRef<LegacyTileCache> m_tileCache;
    RetainPtr<CALayer> m_tileHostLayer;

    IntPoint m_origin;
    IntSize m_tileSize;

    float m_scale;

    typedef UncheckedKeyHashMap<TileIndex, RefPtr<LegacyTileGridTile>> TileMap;
    TileMap m_tiles;

    IntRect m_validBounds;
};

static inline IntPoint topLeft(const IntRect& rect)
{
    return rect.location();
}

static inline IntPoint bottomRight(const IntRect& rect)
{
    return IntPoint(rect.maxX() - 1, rect.maxY() - 1);
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
#endif // TileGrid_h
