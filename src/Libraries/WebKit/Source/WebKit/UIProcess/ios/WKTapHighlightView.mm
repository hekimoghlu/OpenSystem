/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#import "WKTapHighlightView.h"

#if PLATFORM(IOS_FAMILY)

#import <WebCore/PathUtilities.h>
#import <wtf/RetainPtr.h>

@implementation WKTapHighlightView {
    RetainPtr<UIColor> _color;
    float _minimumCornerRadius;
    WebCore::FloatRoundedRect::Radii _cornerRadii;
    Vector<WebCore::FloatRect> _innerFrames;
    Vector<WebCore::FloatQuad> _innerQuads;
}

- (id)initWithFrame:(CGRect)rect
{
    if (self = [super initWithFrame:rect])
        self.layer.needsDisplayOnBoundsChange = YES;
    return self;
}

- (void)cleanUp
{
    _innerFrames.clear();
    _innerQuads.clear();
}

- (void)setColor:(UIColor *)color
{
    _color = color;
}

- (void)setMinimumCornerRadius:(float)radius
{
    _minimumCornerRadius = radius;
}

- (void)setCornerRadii:(WebCore::FloatRoundedRect::Radii&&)radii
{
    _cornerRadii = WTFMove(radii);
}

- (void)setFrames:(Vector<WebCore::FloatRect>&&)frames
{
    [self cleanUp];

    if (frames.isEmpty())
        [self setFrame:CGRectZero];

    bool initialized = false;
    WebCore::FloatRect viewFrame;

    for (auto frame : frames) {
        if (std::exchange(initialized, true))
            viewFrame = WebCore::unionRect(viewFrame, frame);
        else {
            viewFrame = frame;
            viewFrame.inflate(_minimumCornerRadius);
        }
    }

    [super setFrame:viewFrame];

    _innerFrames = WTFMove(frames);
    for (auto& frame : _innerFrames)
        frame.moveBy(-viewFrame.location());

    if (self.layer.needsDisplayOnBoundsChange)
        [self setNeedsDisplay];
}

- (void)setQuads:(Vector<WebCore::FloatQuad>&&)quads boundaryRect:(const WebCore::FloatRect&)boundaryRect
{
    [self cleanUp];

    if (quads.isEmpty())
        [self setFrame:CGRectZero];

    bool initialized = false;
    WebCore::FloatPoint minExtent;
    WebCore::FloatPoint maxExtent;

    for (auto& quad : quads) {
        for (auto controlPoint : std::array { quad.p1(), quad.p2(), quad.p3(), quad.p4() }) {
            if (std::exchange(initialized, true)) {
                minExtent = minExtent.shrunkTo(controlPoint);
                maxExtent = maxExtent.expandedTo(controlPoint);
            } else {
                minExtent = controlPoint;
                maxExtent = controlPoint;
            }
        }
    }

    WebCore::FloatRect viewFrame { minExtent, maxExtent };

    viewFrame.inflate(4 * _minimumCornerRadius);
    viewFrame.intersect(boundaryRect);

    [super setFrame:viewFrame];

    _innerQuads = WTFMove(quads);
    for (auto& quad : _innerQuads)
        quad.move(-viewFrame.x(), -viewFrame.y());

    if (self.layer.needsDisplayOnBoundsChange)
        [self setNeedsDisplay];
}

- (void)setFrame:(CGRect)frame
{
    [self cleanUp];

    [super setFrame:frame];
}

- (void)drawRect:(CGRect)aRect
{
    if (_innerFrames.isEmpty() && _innerQuads.isEmpty()) {
        [_color set];
        [[UIBezierPath bezierPathWithRoundedRect:self.bounds cornerRadius:_minimumCornerRadius] fill];
        return;
    }

    auto path = [UIBezierPath bezierPath];

    if (_innerFrames.size()) {
        auto corePath = WebCore::PathUtilities::pathWithShrinkWrappedRects(_innerFrames, _cornerRadii);
        [path appendPath:[UIBezierPath bezierPathWithCGPath:corePath.platformPath()]];
    } else {
        for (auto& quad : _innerQuads) {
            UIBezierPath *subpath = [UIBezierPath bezierPath];
            [subpath moveToPoint:quad.p1()];
            [subpath addLineToPoint:quad.p2()];
            [subpath addLineToPoint:quad.p3()];
            [subpath addLineToPoint:quad.p4()];
            [subpath closePath];
            [path appendPath:subpath];
        }
    }

    auto context = UIGraphicsGetCurrentContext();
    CGContextSaveGState(context);

    if (!_innerQuads.isEmpty())
        CGContextSetLineWidth(context, 4 * _minimumCornerRadius);

    CGContextSetLineJoin(context, kCGLineJoinRound);

    auto alpha = CGColorGetAlpha([_color CGColor]);

    [[_color colorWithAlphaComponent:1] set];

    CGContextSetAlpha(context, alpha);
    CGContextBeginTransparencyLayer(context, nil);
    CGContextAddPath(context, path.CGPath);
    CGContextDrawPath(context, kCGPathFillStroke);
    CGContextEndTransparencyLayer(context);

    CGContextRestoreGState(context);
}

- (UIView *)hitTest:(CGPoint)point withEvent:(UIEvent *)event
{
    return nil;
}

@end

#endif // PLATFORM(IOS_FAMILY)
