/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#import "WKTextSelectionRect.h"

#if PLATFORM(IOS_FAMILY)
#import "UIKitSPI.h"
#endif
#import <WebCore/SelectionGeometry.h>

#if HAVE(UI_TEXT_SELECTION_RECT_CUSTOM_HANDLE_INFO)

@interface WKTextSelectionRectCustomHandleInfo : UITextSelectionRectCustomHandleInfo
- (instancetype)initWithFloatQuad:(const WebCore::FloatQuad&)quad isHorizontal:(BOOL)isHorizontal;
@end

@implementation WKTextSelectionRectCustomHandleInfo {
    WebCore::FloatQuad _quad;
    BOOL _isHorizontal;
}

- (instancetype)initWithFloatQuad:(const WebCore::FloatQuad&)quad isHorizontal:(BOOL)isHorizontal
{
    if (!(self = [super init]))
        return nil;

    _quad = quad;
    _isHorizontal = isHorizontal;
    return self;
}

- (CGPoint)bottomLeft
{
    return _isHorizontal ? _quad.p4() : _quad.p2();
}

- (CGPoint)topLeft
{
    return _isHorizontal ? _quad.p1() : _quad.p3();
}

- (CGPoint)bottomRight
{
    return _isHorizontal ? _quad.p3() : _quad.p1();
}

- (CGPoint)topRight
{
    return _isHorizontal ? _quad.p2() : _quad.p4();
}

@end

#endif // HAVE(UI_TEXT_SELECTION_RECT_CUSTOM_HANDLE_INFO)

#if PLATFORM(IOS_FAMILY) || HAVE(NSTEXTPLACEHOLDER_RECTS)

@implementation WKTextSelectionRect {
    WebCore::SelectionGeometry _selectionGeometry;
    CGFloat _scaleFactor;
}

- (instancetype)initWithCGRect:(CGRect)rect
{
    WebCore::SelectionGeometry selectionGeometry;
    selectionGeometry.setRect(WebCore::enclosingIntRect(rect));
    return [self initWithSelectionGeometry:WTFMove(selectionGeometry) scaleFactor:1];
}

- (instancetype)initWithSelectionGeometry:(const WebCore::SelectionGeometry&)selectionGeometry scaleFactor:(CGFloat)scaleFactor
{
    if (!(self = [super init]))
        return nil;

    _selectionGeometry = selectionGeometry;
    _scaleFactor = scaleFactor;
    return self;
}

#if PLATFORM(IOS_FAMILY)
- (UIBezierPath *)_path
{
    if (_selectionGeometry.behavior() == WebCore::SelectionRenderingBehavior::CoalesceBoundingRects)
        return nil;

    auto selectionBounds = _selectionGeometry.rect();
    auto quad = _selectionGeometry.quad();
    quad.scale(_scaleFactor);
    quad.move(-selectionBounds.x() * _scaleFactor, -selectionBounds.y() * _scaleFactor);

    auto result = [UIBezierPath bezierPath];
    [result moveToPoint:quad.p1()];
    [result addLineToPoint:quad.p2()];
    [result addLineToPoint:quad.p3()];
    [result addLineToPoint:quad.p4()];
    [result addLineToPoint:quad.p1()];
    [result closePath];
    return result;
}

- (NSWritingDirection)writingDirection
{
    if (_selectionGeometry.direction() == WebCore::TextDirection::LTR)
        return NSWritingDirectionLeftToRight;

    return NSWritingDirectionRightToLeft;
}

- (UITextRange *)range
{
    return nil;
}
#endif

#if HAVE(UI_TEXT_SELECTION_RECT_CUSTOM_HANDLE_INFO)

- (WKTextSelectionRectCustomHandleInfo *)_customHandleInfo
{
    if (_selectionGeometry.behavior() == WebCore::SelectionRenderingBehavior::CoalesceBoundingRects)
        return nil;

    auto scaledQuad = _selectionGeometry.quad();
#if !HAVE(REDESIGNED_TEXT_CURSOR)
    scaledQuad.scale(_scaleFactor);
#endif
    return adoptNS([[WKTextSelectionRectCustomHandleInfo alloc] initWithFloatQuad:scaledQuad isHorizontal:_selectionGeometry.isHorizontal()]).autorelease();
}
#endif // HAVE(UI_TEXT_SELECTION_RECT_CUSTOM_HANDLE_INFO)

- (CGRect)rect
{
    return _selectionGeometry.rect();
}

- (BOOL)containsStart
{
    return _selectionGeometry.containsStart();
}

- (BOOL)containsEnd
{
    return _selectionGeometry.containsEnd();
}

- (BOOL)isVertical
{
    if (_selectionGeometry.behavior() == WebCore::SelectionRenderingBehavior::UseIndividualQuads) {
        // FIXME: Use `!_selectionGeometry.isHorizontal()` for this once rdar://106847585 is fixed.
        return NO;
    }

    return !_selectionGeometry.isHorizontal();
}

@end

#endif
