/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 12, 2023.
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
#import "LegacyTileLayer.h"

#if PLATFORM(IOS_FAMILY)

#import "LegacyTileCache.h"
#import "LegacyTileGrid.h"
#import "WebCoreThread.h"
#import <wtf/SetForScope.h>

using WebCore::LegacyTileCache;
@implementation LegacyTileHostLayer

- (id)initWithTileGrid:(WebCore::LegacyTileGrid*)tileGrid
{
    self = [super init];
    if (!self)
        return nil;
    _tileGrid = tileGrid;
    [self setAnchorPoint:CGPointZero];
    return self;
}

- (id<CAAction>)actionForKey:(NSString *)key
{
    UNUSED_PARAM(key);
    // Disable all default actions
    return nil;
}

- (void)renderInContext:(CGContextRef)context
{
    if (pthread_main_np())
        WebThreadLock();

    CGRect dirtyRect = CGContextGetClipBoundingBox(context);
    Ref tileCache = _tileGrid->tileCache();
    auto useExistingTiles = tileCache->setOverrideVisibleRect(WebCore::FloatRect(dirtyRect));
    if (!useExistingTiles)
        tileCache->doLayoutTiles();

    [super renderInContext:context];

    tileCache->clearOverrideVisibleRect();
    if (!useExistingTiles)
        tileCache->doLayoutTiles();
}
@end

@implementation LegacyTileLayer
@synthesize paintCount = _paintCount;
@synthesize tileGrid = _tileGrid;
@synthesize isRenderingInContext = _isRenderingInContext;

- (void)setNeedsDisplayInRect:(CGRect)rect
{
    // We need to do WebKit layout before painting. Layout may generate new repaint rects and
    // invalidate more tiles, something that is not allowed in drawInContext.
    // Calling setNeedsLayout ensures that layoutSublayers will get called before drawInContext and
    // we do WebKit layout there.
    [self setNeedsLayout];
    [super setNeedsDisplayInRect:rect];
}

- (void)layoutSublayers
{
    if (pthread_main_np())
        WebThreadLock();
    // This may trigger WebKit layout and generate more repaint rects.
    if (_tileGrid)
        _tileGrid->protectedTileCache()->prepareToDraw();
}

- (void)renderInContext:(CGContextRef)context
{
    SetForScope change(_isRenderingInContext, YES);
    [super renderInContext:context];
}

- (void)drawInContext:(CGContextRef)context
{
    // Bugs in clients or other frameworks may cause tile invalidation from within a CA commit.
    // In that case we maybe left with dirty tiles that have display still pending. Some future
    // commit will flush such tiles and they will get painted without holding the web lock.
    // rdar://problem/21149759
    // Still assert as the condition is not normal and may cause graphical glitches.
    ASSERT(WebThreadIsLockedOrDisabled());
    if (pthread_main_np())
        WebThreadLock();

    if (_tileGrid)
        _tileGrid->protectedTileCache()->drawLayer(self, context, self.isRenderingInContext ? LegacyTileCache::DrawingFlags::Snapshotting : LegacyTileCache::DrawingFlags::None);
}

- (id<CAAction>)actionForKey:(NSString *)key
{
    UNUSED_PARAM(key);
    // Disable all default actions
    return nil;
}

@end

#endif // PLATFORM(IOS_FAMILY)
