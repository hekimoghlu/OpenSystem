/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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

#if PLATFORM(IOS_FAMILY)

#include "FloatRect.h"
#include "IntRect.h"
#include "Timer.h"
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/RetainPtr.h>
#include <wtf/Threading.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS CALayer;
OBJC_CLASS LegacyTileCacheTombstone;
OBJC_CLASS LegacyTileLayer;
OBJC_CLASS WAKWindow;

namespace WebCore {

class Color;
class LegacyTileGrid;

class LegacyTileCache : public CanMakeWeakPtr<LegacyTileCache> {
    WTF_MAKE_NONCOPYABLE(LegacyTileCache);
public:
    LegacyTileCache(WAKWindow *);
    ~LegacyTileCache();

    void ref() const;
    void deref() const;

    CGFloat screenScale() const;

    void setNeedsDisplay();
    void setNeedsDisplayInRect(const IntRect&);
    
    void layoutTiles();
    void layoutTilesNow();
    void layoutTilesNowForRect(const IntRect&);
    void removeAllNonVisibleTiles();
    void removeAllTiles();
    void removeForegroundTiles();

    // If 'contentReplacementImage' is not NULL, drawLayer() draws
    // contentReplacementImage instead of the page content. We assume the
    // image is to be drawn at the origin and scaled to match device pixels.
    void setContentReplacementImage(RetainPtr<CGImageRef>);
    RetainPtr<CGImageRef> contentReplacementImage() const;

    WEBCORE_EXPORT void setTileBordersVisible(bool);
    bool tileBordersVisible() const { return m_tileBordersVisible; }

    WEBCORE_EXPORT void setTilePaintCountersVisible(bool);
    bool tilePaintCountersVisible() const { return m_tilePaintCountersVisible; }

    void setAcceleratedDrawingEnabled(bool enabled) { m_acceleratedDrawingEnabled = enabled; }
    bool acceleratedDrawingEnabled() const { return m_acceleratedDrawingEnabled; }

    void setKeepsZoomedOutTiles(bool);
    bool keepsZoomedOutTiles() const { return m_keepsZoomedOutTiles; }

    void setZoomedOutScale(float);
    float zoomedOutScale() const;
    
    void setCurrentScale(float);
    float currentScale() const;
    
    bool tilesOpaque() const { return m_tilesOpaque; }
    void setTilesOpaque(bool);
    
    enum TilingMode {
        Normal,
        Minimal,
        Panning,
        Zooming,
        Disabled,
        ScrollToTop
    };
    TilingMode tilingMode() const { return m_tilingMode; }
    void setTilingMode(TilingMode);

    enum TilingDirection {
        TilingDirectionUp,
        TilingDirectionDown,
        TilingDirectionLeft,
        TilingDirectionRight,
    };
    void setTilingDirection(TilingDirection tilingDirection) { m_tilingDirection = tilingDirection; }
    TilingDirection tilingDirection() const { return m_tilingDirection; }

    void hostLayerSizeChanged();

    WEBCORE_EXPORT static void setLayerPoolCapacity(unsigned);
    WEBCORE_EXPORT static void drainLayerPool();

    // Logging
    void dumpTiles();

    // Internal
    void doLayoutTiles();
    
    enum class DrawingFlags { None, Snapshotting };
    void drawLayer(LegacyTileLayer *, CGContextRef, DrawingFlags);
    void prepareToDraw();
    void finishedCreatingTiles(bool didCreateTiles, bool createMore);
    FloatRect visibleRectInLayer(CALayer *) const;
    CALayer* hostLayer() const;
    unsigned tileCapacityForGrid(LegacyTileGrid*);
    Color colorForGridTileBorder(LegacyTileGrid*) const;
    bool setOverrideVisibleRect(const FloatRect&);
    void clearOverrideVisibleRect() { m_overrideVisibleRect = std::nullopt; }

    void doPendingRepaints();

    bool isSpeculativeTileCreationEnabled() const { return m_isSpeculativeTileCreationEnabled; }
    void setSpeculativeTileCreationEnabled(bool);
    
    enum SynchronousTileCreationMode { CoverVisibleOnly, CoverSpeculative };

    bool tileControllerShouldUseLowScaleTiles() const { return m_tileControllerShouldUseLowScaleTiles; } 
    void setTileControllerShouldUseLowScaleTiles(bool flag) { m_tileControllerShouldUseLowScaleTiles = flag; } 

private:
    LegacyTileGrid* activeTileGrid() const;
    LegacyTileGrid* inactiveTileGrid() const;

    void updateTilingMode();
    bool isTileInvalidationSuspended() const;
    bool isTileCreationSuspended() const;
    void flushSavedDisplayRects();
    void invalidateTiles(const IntRect& dirtyRect);
    void setZoomedOutScaleInternal(float);
    void commitScaleChange();
    void bringActiveTileGridToFront();
    void adjustTileGridTransforms();
    void removeAllNonVisibleTilesInternal();
    void createTilesInActiveGrid(SynchronousTileCreationMode);
    void scheduleRenderingUpdateForPendingRepaint();

    void tileCreationTimerFired();

    void drawReplacementImage(LegacyTileLayer *, CGContextRef, CGImageRef);
    void drawWindowContent(LegacyTileLayer *, CGContextRef, CGRect dirtyRect, DrawingFlags);

    WAKWindow *m_window { nullptr };

    RetainPtr<CGImageRef> m_contentReplacementImage;

    // Ensure there are no async calls on a dead tile cache.
    RetainPtr<LegacyTileCacheTombstone> m_tombstone;

    std::optional<FloatRect> m_overrideVisibleRect;

    IntSize m_tileSize { 512, 512 };
    
    TilingMode m_tilingMode { Normal };
    TilingDirection m_tilingDirection { TilingDirectionDown };
    
    bool m_keepsZoomedOutTiles { false };
    bool m_hasPendingLayoutTiles { false };
    bool m_hasPendingUpdateTilingMode { false };
    bool m_tilesOpaque { true };
    bool m_tileBordersVisible { false };
    bool m_tilePaintCountersVisible { false };
    bool m_acceleratedDrawingEnabled { false };
    bool m_isSpeculativeTileCreationEnabled { true };
    bool m_tileControllerShouldUseLowScaleTiles { false };
    bool m_didCallWillStartScrollingOrZooming { false };
    
    std::unique_ptr<LegacyTileGrid> m_zoomedOutTileGrid;
    std::unique_ptr<LegacyTileGrid> m_zoomedInTileGrid;

    Timer m_tileCreationTimer;

    Vector<IntRect> m_savedDisplayRects;

    float m_currentScale { 1 };

    float m_pendingScale { 0 };
    float m_pendingZoomedOutScale { 0 };

    mutable Lock m_tileMutex;
    mutable Lock m_savedDisplayRectMutex;
    mutable Lock m_contentReplacementImageMutex;
};

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
