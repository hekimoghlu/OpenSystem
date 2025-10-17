/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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

#include "IntPoint.h"
#include "PlatformLayerIdentifier.h"
#include "TileGridIdentifier.h"
#include <wtf/CheckedRef.h>
#include <wtf/MonotonicTime.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class TiledBackingClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::TiledBackingClient> : std::true_type { };
}

namespace WebCore {

class FloatPoint;
class FloatRect;
class FloatSize;
class IntRect;
class IntSize;
class PlatformCALayer;

struct VelocityData;

enum ScrollingModeIndication {
    SynchronousScrollingBecauseOfLackOfScrollingCoordinatorIndication,
    SynchronousScrollingBecauseOfStyleIndication,
    SynchronousScrollingBecauseOfEventHandlersIndication,
    AsyncScrollingIndication
};

enum class TiledBackingScrollability : uint8_t {
    NotScrollable           = 0,
    HorizontallyScrollable  = 1 << 0,
    VerticallyScrollable    = 1 << 1
};

enum class TileRevalidationType : uint8_t {
    Partial,
    Full
};

using TileIndex = IntPoint;
class TiledBacking;

class TiledBackingClient : public CanMakeWeakPtr<TiledBackingClient> {
public:
    virtual ~TiledBackingClient() = default;

    // paintDirtyRect is in the same coordinate system as tileClip.
    virtual void willRepaintTile(TiledBacking&, TileGridIdentifier, TileIndex, const FloatRect& tileClip, const FloatRect& paintDirtyRect) = 0;
    virtual void willRemoveTile(TiledBacking&, TileGridIdentifier, TileIndex) = 0;
    virtual void willRepaintAllTiles(TiledBacking&, TileGridIdentifier) = 0;

    // The client will not receive `willRepaintTile()` for tiles needing display as part of a revalidation.
    virtual void willRevalidateTiles(TiledBacking&, TileGridIdentifier, TileRevalidationType) = 0;
    virtual void didRevalidateTiles(TiledBacking&, TileGridIdentifier, TileRevalidationType, const UncheckedKeyHashSet<TileIndex>& tilesNeedingDisplay) = 0;

    virtual void didAddGrid(TiledBacking&, TileGridIdentifier) = 0;
    virtual void willRemoveGrid(TiledBacking&, TileGridIdentifier) = 0;

    virtual void coverageRectDidChange(TiledBacking&, const FloatRect&) = 0;

    virtual void willRepaintTilesAfterScaleFactorChange(TiledBacking&, TileGridIdentifier) = 0;
    virtual void didRepaintTilesAfterScaleFactorChange(TiledBacking&, TileGridIdentifier) = 0;
};


class TiledBacking : public CanMakeCheckedPtr<TiledBacking> {
    WTF_MAKE_TZONE_ALLOCATED(TiledBacking);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TiledBacking);
public:
    virtual ~TiledBacking() = default;

    virtual PlatformLayerIdentifier layerIdentifier() const = 0;

    virtual void setClient(TiledBackingClient*) = 0;

    // Note that the grids switch or change over time.
    virtual TileGridIdentifier primaryGridIdentifier() const = 0;
    // There can be a secondary grid when setZoomedOutContentsScale() has been called.
    virtual std::optional<TileGridIdentifier> secondaryGridIdentifier() const = 0;

    virtual void setVisibleRect(const FloatRect&) = 0;
    virtual FloatRect visibleRect() const = 0;

    // Only used to update the tile coverage map.
    virtual void setLayoutViewportRect(std::optional<FloatRect>) = 0;

    virtual void setCoverageRect(const FloatRect&) = 0;
    virtual FloatRect coverageRect() const = 0;
    virtual bool tilesWouldChangeForCoverageRect(const FloatRect&) const = 0;

    virtual void setTiledScrollingIndicatorPosition(const FloatPoint&) = 0;
    virtual void setTopContentInset(float) = 0;

    virtual void setVelocity(const VelocityData&) = 0;

    virtual void setTileSizeUpdateDelayDisabledForTesting(bool) = 0;
    
    using Scrollability = TiledBackingScrollability;
    virtual void setScrollability(OptionSet<Scrollability>) = 0;

    virtual void prepopulateRect(const FloatRect&) = 0;

    virtual void setIsInWindow(bool) = 0;
    virtual bool isInWindow() const = 0;

    enum {
        CoverageForVisibleArea = 0,
        CoverageForVerticalScrolling = 1 << 0,
        CoverageForHorizontalScrolling = 1 << 1,
        CoverageForScrolling = CoverageForVerticalScrolling | CoverageForHorizontalScrolling
    };
    typedef unsigned TileCoverage;

    virtual void setTileCoverage(TileCoverage) = 0;
    virtual TileCoverage tileCoverage() const = 0;

    virtual FloatRect adjustTileCoverageRect(const FloatRect& coverageRect, const FloatRect& previousVisibleRect, const FloatRect& currentVisibleRect, bool sizeChanged) = 0;
    virtual FloatRect adjustTileCoverageRectForScrolling(const FloatRect& coverageRect, const FloatSize& newSize, const FloatRect& previousVisibleRect, const FloatRect& currentVisibleRect, float contentsScale) = 0;

    virtual void willStartLiveResize() = 0;
    virtual void didEndLiveResize() = 0;

    virtual IntSize tileSize() const = 0;
    // The returned rect is in the same coordinate space as the tileClip rect argument to willRepaintTile().
    virtual FloatRect rectForTile(TileIndex) const = 0;

    virtual void revalidateTiles() = 0;

    virtual void setScrollingPerformanceTestingEnabled(bool) = 0;
    virtual bool scrollingPerformanceTestingEnabled() const = 0;
    
    virtual double retainedTileBackingStoreMemory() const = 0;

    virtual void setHasMargins(bool marginTop, bool marginBottom, bool marginLeft, bool marginRight) = 0;
    virtual void setMarginSize(int) = 0;
    virtual bool hasMargins() const = 0;
    virtual bool hasHorizontalMargins() const = 0;
    virtual bool hasVerticalMargins() const = 0;

    virtual int topMarginHeight() const = 0;
    virtual int bottomMarginHeight() const = 0;
    virtual int leftMarginWidth() const = 0;
    virtual int rightMarginWidth() const = 0;

    // This is the scale used to compute tile sizes; it's contentScale / deviceScaleFactor.
    virtual float tilingScaleFactor() const  = 0;

    virtual void setZoomedOutContentsScale(float) = 0;
    virtual float zoomedOutContentsScale() const = 0;

    // Includes margins.
    virtual IntRect bounds() const = 0;
    virtual IntRect boundsWithoutMargin() const = 0;

    // Exposed for testing
    virtual IntRect tileCoverageRect() const = 0;
    virtual IntRect tileGridExtent() const = 0;
    virtual void setScrollingModeIndication(ScrollingModeIndication) = 0;

#if USE(CA)
    virtual PlatformCALayer* tiledScrollingIndicatorLayer() = 0;
#endif
};

} // namespace WebCore
