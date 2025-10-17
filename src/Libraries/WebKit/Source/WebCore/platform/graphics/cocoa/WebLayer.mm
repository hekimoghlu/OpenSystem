/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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
#import "WebLayer.h"

#import "GraphicsContextCG.h"
#import "GraphicsLayerCA.h"
#import "PlatformCALayer.h"
#import <QuartzCore/QuartzCore.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/SetForScope.h>

#if PLATFORM(IOS_FAMILY)
#import "WAKWindow.h"
#import "WKGraphics.h"
#import "WebCoreThread.h"
#endif

#if PLATFORM(IOS_FAMILY)
@interface WebLayer(Private)
- (void)drawScaledContentsInContext:(CGContextRef)context;
@end
#endif

@implementation WebLayer

- (void)drawInContext:(CGContextRef)context
{
    auto layer = WebCore::PlatformCALayer::platformCALayerForLayer((__bridge void*)self);
    if (layer) {
        WebCore::GraphicsContextCG graphicsContext(context, WebCore::GraphicsContextCG::CGContextFromCALayer);
        WebCore::PlatformCALayer::RepaintRectList rectsToPaint = WebCore::PlatformCALayer::collectRectsToPaint(graphicsContext, layer.get());
        OptionSet<WebCore::GraphicsLayerPaintBehavior> paintBehavior;
        if (self.isRenderingInContext)
            paintBehavior.add(WebCore::GraphicsLayerPaintBehavior::ForceSynchronousImageDecode);
        WebCore::PlatformCALayer::drawLayerContents(graphicsContext, layer.get(), rectsToPaint, paintBehavior);
    }
}

@end // implementation WebLayer

@implementation WebSimpleLayer

@synthesize isRenderingInContext = _isRenderingInContext;

- (void)renderInContext:(CGContextRef)context
{
    SetForScope change(_isRenderingInContext, YES);
    [super renderInContext:context];
}

- (id<CAAction>)actionForKey:(NSString *)key
{
    // Fix for <rdar://problem/9015675>: Force the layer content to be updated when the tree is reparented.
    if ([key isEqualToString:@"onOrderIn"])
        [self reloadValueForKeyPath:@"contents"];

    return nil;
}

- (void)setNeedsDisplay
{
    auto layer = WebCore::PlatformCALayer::platformCALayerForLayer((__bridge void*)self);
    if (!layer || !layer->owner())
        return;
    if (!layer->owner()->platformCALayerDrawsContent() && !layer->owner()->platformCALayerDelegatesDisplay(layer.get()))
        return;
    [super setNeedsDisplay];
}

- (void)setNeedsDisplayInRect:(CGRect)dirtyRect
{
    auto platformLayer = WebCore::PlatformCALayer::platformCALayerForLayer((__bridge void*)self);
    if (!platformLayer) {
        [super setNeedsDisplayInRect:dirtyRect];
        return;
    }

    if (WebCore::PlatformCALayerClient* layerOwner = platformLayer->owner()) {
        if (layerOwner->platformCALayerDrawsContent() || layerOwner->platformCALayerDelegatesDisplay(platformLayer.get())) {
            [super setNeedsDisplayInRect:dirtyRect];

            if (layerOwner->platformCALayerShowRepaintCounter(platformLayer.get())) {
                CGRect bounds = [self bounds];
                CGRect indicatorRect = CGRectMake(bounds.origin.x, bounds.origin.y, 52, 27);
                [super setNeedsDisplayInRect:indicatorRect];
            }
        }
    }
}

- (void)display
{
#if PLATFORM(IOS_FAMILY)
    if (pthread_main_np())
        WebThreadLock();
#endif
    ASSERT(isMainThread());
    auto layer = WebCore::PlatformCALayer::platformCALayerForLayer((__bridge void*)self);
    WebCore::PlatformCALayerClient* owner = layer ? layer->owner() : nullptr;
    if (owner && owner->platformCALayerDelegatesDisplay(layer.get()))
        owner->platformCALayerLayerDisplay(layer.get());
    else
        [super display];
    if (owner)
        owner->platformCALayerLayerDidDisplay(layer.get());
}

- (void)drawInContext:(CGContextRef)context
{
#if PLATFORM(IOS_FAMILY)
    if (pthread_main_np())
        WebThreadLock();
#endif
    ASSERT(isMainThread());
    auto layer = WebCore::PlatformCALayer::platformCALayerForLayer((__bridge void*)self);
    if (layer && layer->owner()) {
        WebCore::GraphicsContextCG graphicsContext(context, WebCore::GraphicsContextCG::CGContextFromCALayer);
        WebCore::FloatRect clipBounds = CGContextGetClipBoundingBox(context);
        OptionSet<WebCore::GraphicsLayerPaintBehavior> paintBehavior;
        if (self.isRenderingInContext)
            paintBehavior.add(WebCore::GraphicsLayerPaintBehavior::ForceSynchronousImageDecode);
        layer->owner()->platformCALayerPaintContents(layer.get(), graphicsContext, clipBounds, paintBehavior);
    }
}

@end // implementation WebSimpleLayer

#ifndef NDEBUG

@implementation CALayer(ExtendedDescription)

- (NSString*)_descriptionWithPrefix:(NSString*)inPrefix
{
    CGRect aBounds = [self bounds];
    CGPoint aPos = [self position];

    NSString* selfString = [NSString stringWithFormat:@"%@<%@ 0x%p> \"%@\" bounds(%.1f, %.1f, %.1f, %.1f) pos(%.1f, %.1f), sublayers=%lu masking=%d",
        inPrefix,
        [self class],
        self,
        [self name],
        aBounds.origin.x, aBounds.origin.y, aBounds.size.width, aBounds.size.height,
        aPos.x, aPos.y,
        static_cast<unsigned long>([[self sublayers] count]),
        [self masksToBounds]];

    NSMutableString* curDesc = [NSMutableString stringWithString:selfString];

    if ([[self sublayers] count] > 0)
        [curDesc appendString:@"\n"];

    NSString* sublayerPrefix = [inPrefix stringByAppendingString:@"\t"];

    NSEnumerator* sublayersEnum = [[self sublayers] objectEnumerator];
    CALayer* curLayer;
    while ((curLayer = [sublayersEnum nextObject]))
        [curDesc appendString:[curLayer _descriptionWithPrefix:sublayerPrefix]];

    if (![[self sublayers] count])
        [curDesc appendString:@"\n"];

    if (CALayer *mask = [self mask]) {
        [curDesc appendString:@"mask: "];
        [curDesc appendString:[mask _descriptionWithPrefix:sublayerPrefix]];
    }

    return curDesc;
}

- (NSString*)extendedDescription
{
    return [self _descriptionWithPrefix:@""];
}

@end // implementation WebLayer(ExtendedDescription)

#endif // NDEBUG
