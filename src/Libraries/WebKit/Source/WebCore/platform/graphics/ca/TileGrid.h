/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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

#include "IntPointHash.h"
#include "IntRect.h"
#include "PlatformCALayerClient.h"
#include "TileGridIdentifier.h"
#include "Timer.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Deque.h>
#include <wtf/HashCountedSet.h>
#include <wtf/HashMap.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>

#if USE(CG)
typedef struct CGContext *CGContextRef;
#endif

namespace WebCore {

class GraphicsContext;
class PlatformCALayer;
class TileController;

using TileIndex = IntPoint;

class TileGrid final : public PlatformCALayerClient, public CanMakeCheckedPtr<TileGrid> {
    WTF_MAKE_TZONE_ALLOCATED(TileGrid);
    WTF_MAKE_NONCOPYABLE(TileGrid);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TileGrid);
public:
    explicit TileGrid(TileController&);
    ~TileGrid();

    TileGridIdentifier identifier() const { return m_identifier; }

#if USE(CA)
    PlatformCALayer& containerLayer() { return m_containerLayer; }
#endif

    void setIsZoomedOutTileGrid(bool);

    void setScale(float);
    float scale() const { return m_scale; }

    void setNeedsDisplay();
    void setNeedsDisplayInRect(const IntRect&);
    void dropTilesInRect(const IntRect&);

    void updateTileLayerProperties();

    bool prepopulateRect(const FloatRect&);

    enum ValidationPolicyFlag : uint8_t {
        PruneSecondaryTiles = 1 << 0,
        UnparentAllTiles    = 1 << 1
    };
    void revalidateTiles(OptionSet<ValidationPolicyFlag> = { });

    bool tilesWouldChangeForCoverageRect(const FloatRect&) const;

    IntRect tileCoverageRect() const;
    IntRect extent() const;
    
    IntSize tileSize() const { return m_tileSize; }
    FloatRect rectForTile(TileIndex) const;

    double retainedTileBackingStoreMemory() const;
    unsigned blankPixelCount() const;

#if USE(CG)
    void drawTileMapContents(CGContextRef, CGRect layerBounds) const;
#endif

#if PLATFORM(IOS_FAMILY)
    unsigned numberOfUnparentedTiles() const { return m_cohortList.size(); }
    void removeUnparentedTilesNow();
#endif

    using TileCohort = unsigned;
    static constexpr TileCohort visibleTileCohort = std::numeric_limits<TileCohort>::max();

    struct TileInfo {
        RefPtr<PlatformCALayer> layer;
        TileCohort cohort { visibleTileCohort };
        bool hasStaleContent { false };
    };

private:
    void setTileNeedsDisplayInRect(const TileIndex&, TileInfo&, const IntRect& repaintRectInTileCoords, const IntRect& coverageRectInTileCoords);

    IntRect rectForTileIndex(const TileIndex&) const;
    bool getTileIndexRangeForRect(const IntRect&, TileIndex& topLeft, TileIndex& bottomRight) const;

    enum class CoverageType { PrimaryTiles, SecondaryTiles };
    IntRect ensureTilesForRect(const FloatRect&, UncheckedKeyHashSet<TileIndex>& tilesNeedingDisplay, CoverageType);

    struct TileCohortInfo {
        TileCohort cohort;
        MonotonicTime creationTime;
        TileCohortInfo(TileCohort inCohort, MonotonicTime inTime)
            : cohort(inCohort)
            , creationTime(inTime)
        { }

        Seconds timeUntilExpiration();
    };

    void removeAllTiles();
    void removeAllSecondaryTiles();
    void removeTilesInCohort(TileCohort);

    void scheduleCohortRemoval();
    void cohortRemovalTimerFired();
    TileCohort nextTileCohort() const;
    void startedNewCohort(TileCohort);
    TileCohort newestTileCohort() const;
    TileCohort oldestTileCohort() const;

    void removeTiles(const Vector<TileIndex>& toRemove);

    // PlatformCALayerClient
    PlatformLayerIdentifier platformCALayerIdentifier() const override;
    void platformCALayerPaintContents(PlatformCALayer*, GraphicsContext&, const FloatRect&, OptionSet<GraphicsLayerPaintBehavior>) override;
    bool platformCALayerShowDebugBorders() const override;
    bool platformCALayerShowRepaintCounter(PlatformCALayer*) const override;
    int platformCALayerRepaintCount(PlatformCALayer*) const override;
    int platformCALayerIncrementRepaintCount(PlatformCALayer*) override;
    bool platformCALayerContentsOpaque() const override;
    bool platformCALayerDrawsContent() const override { return true; }
    float platformCALayerDeviceScaleFactor() const override;
    bool isUsingDisplayListDrawing(PlatformCALayer*) const override;
    bool platformCALayerNeedsPlatformContext(const PlatformCALayer*) const override;

    TileGridIdentifier m_identifier;
    CheckedRef<TileController> m_controller;
#if USE(CA)
    Ref<PlatformCALayer> m_containerLayer;
#endif

    UncheckedKeyHashMap<TileIndex, TileInfo> m_tiles;

    IntRect m_primaryTileCoverageRect;
    Vector<FloatRect> m_secondaryTileCoverageRects;

    Deque<TileCohortInfo> m_cohortList;

    Timer m_cohortRemovalTimer;

    HashCountedSet<PlatformCALayer*> m_tileRepaintCounts;
    
    IntSize m_tileSize;

    float m_scale { 1 };
    std::optional<float> m_scaleAtLastRevalidation;
};

}
