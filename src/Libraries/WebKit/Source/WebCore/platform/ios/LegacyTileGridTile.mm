/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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
#import "config.h"
#import "LegacyTileGridTile.h"

#if PLATFORM(IOS_FAMILY)

#import "Color.h"
#import "ContentsFormatCocoa.h"
#import "IOSurface.h"
#import "LegacyTileCache.h"
#import "LegacyTileGrid.h"
#import "LegacyTileLayer.h"
#import "LegacyTileLayerPool.h"
#import "PlatformScreen.h"
#import "WAKWindow.h"
#import <algorithm>
#import <functional>
#import <pal/spi/cocoa/QuartzCoreSPI.h>

namespace WebCore {

#if LOG_TILING
static int totalTileCount;
#endif

LegacyTileGridTile::LegacyTileGridTile(LegacyTileGrid* tileGrid, const IntRect& tileRect)
    : m_tileGrid(tileGrid)
    , m_rect(tileRect)
{
    ASSERT(!tileRect.isEmpty());
    IntSize pixelSize(m_rect.size());
    Ref tileCache = m_tileGrid->tileCache();
    const CGFloat screenScale = tileCache->screenScale();
    pixelSize.scale(screenScale);
    m_tileLayer = LegacyTileLayerPool::sharedPool()->takeLayerWithSize(pixelSize);
    if (!m_tileLayer) {
#if LOG_TILING
        NSLog(@"unable to reuse layer with size %d x %d, creating one", pixelSize.width(), pixelSize.height());
#endif
        m_tileLayer = adoptNS([[LegacyTileLayer alloc] init]);
    }
    LegacyTileLayer* layer = m_tileLayer.get();

    if (NSString *formatString = contentsFormatString(screenContentsFormat()))
        layer.contentsFormat = formatString;

    [layer setTileGrid:tileGrid];
    [layer setOpaque:tileCache->tilesOpaque()];
    [layer setEdgeAntialiasingMask:0];
    [layer setNeedsLayoutOnGeometryChange:NO];
    [layer setContentsScale:screenScale];
    [layer setDrawsAsynchronously:tileCache->acceleratedDrawingEnabled()];

    // Host layer may have other sublayers. Keep the tile layers at the beginning of the array
    // so they are painted behind everything else.
    [tileGrid->tileHostLayer() insertSublayer:layer atIndex:tileGrid->tileCount()];
    [layer setFrame:m_rect];
    invalidateRect(m_rect);
    showBorder(tileCache->tileBordersVisible());

#if LOG_TILING
    ++totalTileCount;
    NSLog(@"new Tile (%d,%d) %d %d, count %d", tileRect.x(), tileRect.y(), tileRect.width(), tileRect.height(), totalTileCount);
#endif
}

LegacyTileGridTile::~LegacyTileGridTile() 
{
    [tileLayer() setTileGrid:0];
    [tileLayer() removeFromSuperlayer];
    LegacyTileLayerPool::sharedPool()->addLayer(tileLayer());
#if LOG_TILING
    --totalTileCount;
    NSLog(@"delete Tile (%d,%d) %d %d, count %d", m_rect.x(), m_rect.y(), m_rect.width(), m_rect.height(), totalTileCount);
#endif
}

void LegacyTileGridTile::invalidateRect(const IntRect& windowDirtyRect)
{
    IntRect dirtyRect = intersection(windowDirtyRect, m_rect);
    if (dirtyRect.isEmpty())
        return;
    dirtyRect.move(IntPoint() - m_rect.location());
    [tileLayer() setNeedsDisplayInRect:dirtyRect];

    if (m_tileGrid->tileCache().tilePaintCountersVisible())
        [tileLayer() setNeedsDisplayInRect:CGRectMake(0, 0, 46, 25)];
}

void LegacyTileGridTile::setRect(const IntRect& tileRect)
{
    if (m_rect == tileRect)
        return;
    m_rect = tileRect;
    LegacyTileLayer* layer = m_tileLayer.get();
    [layer setFrame:m_rect];
    [layer setNeedsDisplay];
}

void LegacyTileGridTile::showBorder(bool flag)
{
    LegacyTileLayer* layer = m_tileLayer.get();
    if (flag) {
        [layer setBorderColor:cachedCGColor(m_tileGrid->protectedTileCache()->colorForGridTileBorder(m_tileGrid)).get()];
        [layer setBorderWidth:0.5f];
    } else {
        [layer setBorderColor:nil];
        [layer setBorderWidth:0];
    }
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
