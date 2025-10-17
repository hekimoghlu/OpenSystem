/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
#import "WebTiledBackingLayer.h"

#import "IntRect.h"
#import "TileController.h"
#import <wtf/MainThread.h>

@implementation WebTiledBackingLayer

- (id)init
{
    self = [super init];
    if (!self)
        return nil;

#ifndef NDEBUG
    [self setName:@"WebTiledBackingLayer"];
#endif
    return self;
}

- (void)dealloc
{
    ASSERT(!_tileController);

    [super dealloc];
}

- (WebCore::TileController*)createTileController:(WebCore::PlatformCALayer*)rootLayer
{
    ASSERT(!_tileController);
    _tileController = makeUnique<WebCore::TileController>(rootLayer);

    // Sync the underlying layer with the controller's scale, and keep the rasterization scale the same, as PlatformCALayerCocoa does.
    CGFloat initialScale = _tileController->contentsScale();
    [super setContentsScale:initialScale];
    [super setRasterizationScale:initialScale];

    return _tileController.get();
}

- (id<CAAction>)actionForKey:(NSString *)key
{
    UNUSED_PARAM(key);
    
    // Disable all animations.
    return nil;
}

- (void)setBounds:(CGRect)bounds
{
    [super setBounds:bounds];

    _tileController->tileCacheLayerBoundsChanged();
}

- (void)setOpaque:(BOOL)opaque
{
    _tileController->setTilesOpaque(opaque);
}

- (BOOL)isOpaque
{
    return _tileController ? _tileController->tilesAreOpaque() : NO;
}

- (void)setNeedsDisplay
{
    _tileController->setNeedsDisplay();
}

- (void)setNeedsDisplayInRect:(CGRect)rect
{
    _tileController->setNeedsDisplayInRect(WebCore::enclosingIntRect(rect));
}

- (void)setDrawsAsynchronously:(BOOL)acceleratesDrawing
{
    _tileController->setAcceleratesDrawing(acceleratesDrawing);
}

- (BOOL)drawsAsynchronously
{
    return _tileController ? _tileController->acceleratesDrawing() : NO;
}

- (void)setContentsFormat:(WebCore::ContentsFormat)contentsFormat
{
    _tileController->setContentsFormat(contentsFormat);
}

- (WebCore::ContentsFormat)contentsFormat
{
    return _tileController->contentsFormat();
}

- (void)setContentsScale:(CGFloat)contentsScale
{
    [super setContentsScale:contentsScale];
    _tileController->setContentsScale(contentsScale);
}

- (CGFloat)contentsScale
{
    return _tileController ? _tileController->contentsScale() : 1;
}

- (WebCore::TiledBacking*)tiledBacking
{
    return _tileController.get();
}

- (void)invalidate
{
    ASSERT(isMainThread());
    ASSERT(_tileController);
    _tileController = nullptr;
}

- (void)setBorderColor:(CGColorRef)borderColor
{
    _tileController->setTileDebugBorderColor(WebCore::roundAndClampToSRGBALossy(borderColor));
}

- (void)setBorderWidth:(CGFloat)borderWidth
{
    // Tiles adjoin, so halve the border width.
    _tileController->setTileDebugBorderWidth(borderWidth / 2);
}

@end
