/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
#import "SwitchMacUtilities.h"

#if PLATFORM(MAC)

#import "GraphicsContext.h"
#import "ImageBuffer.h"
#import "LocalCurrentGraphicsContext.h"
#import <pal/spi/mac/CoreUISPI.h>
#import <pal/spi/mac/NSAppearanceSPI.h>

namespace WebCore::SwitchMacUtilities {

IntSize cellSize(NSControlSize controlSize)
{
    static const std::array<IntSize, 4> sizes =
    {
        IntSize { 38, 22 },
        IntSize { 32, 18 },
        IntSize { 26, 15 },
        IntSize { 38, 22 }
    };
    return sizes[controlSize];
}

FloatSize visualCellSize(IntSize size, const ControlStyle& style)
{
    if (style.states.contains(ControlStyle::State::VerticalWritingMode))
        size = size.transposedSize();
    size.scale(style.zoomFactor);
    return size;
}

IntOutsets cellOutsets(NSControlSize controlSize)
{
    static const std::array margins {
        // top right bottom left
        IntOutsets { 2, 2, 1, 2 },
        IntOutsets { 2, 2, 1, 2 },
        IntOutsets { 1, 1, 0, 1 },
        IntOutsets { 2, 2, 1, 2 },
    };
    return margins[controlSize];
}

IntOutsets visualCellOutsets(NSControlSize controlSize, bool isVertical)
{
    auto outsets = cellOutsets(controlSize);
    if (isVertical)
        outsets = { outsets.left(), outsets.top(), outsets.right(), outsets.bottom() };
    return outsets;
}

FloatRect rectForBounds(const FloatRect& bounds)
{
    ASSERT_NOT_IMPLEMENTED_YET();
    return bounds;
}

NSString *coreUISizeForControlSize(const NSControlSize controlSize)
{
    if (controlSize == NSControlSizeMini)
        return (__bridge NSString *)kCUISizeMini;
    if (controlSize == NSControlSizeSmall)
        return (__bridge NSString *)kCUISizeSmall;
    return (__bridge NSString *)kCUISizeRegular;
}

float easeInOut(const float progress)
{
    return -2.0f * pow(progress, 3.0f) + 3.0f * pow(progress, 2.0f);
}

FloatRect rectWithTransposedSize(const FloatRect& rect, bool isVertical)
{
    auto logicalRect = rect;
    if (isVertical)
        logicalRect.setSize(logicalRect.size().transposedSize());
    return logicalRect;
}

FloatRect trackRectForBounds(const FloatRect& bounds, const FloatSize& size)
{
    auto offsetY = std::max(((bounds.height() - size.height()) / 2.0f), 0.0f);
    return { FloatPoint { bounds.x(), bounds.y() + offsetY }, size };
}

void rotateContextForVerticalWritingMode(GraphicsContext& context, const FloatRect& rect)
{
    context.translate(rect.height(), 0);
    context.translate(rect.location());
    context.rotate(piOverTwoFloat);
    context.translate(-rect.location());
}

RefPtr<ImageBuffer> trackMaskImage(GraphicsContext& context, FloatSize trackRectSize, float deviceScaleFactor, bool isRTL, NSString *coreUISize)
{
    auto drawingTrackRect = NSMakeRect(0, 0, trackRectSize.width(), trackRectSize.height());
    auto maskImage = context.createImageBuffer(trackRectSize, deviceScaleFactor);
    if (!maskImage)
        return nullptr;

    auto cgContext = maskImage->context().platformContext();

    auto coreUIDirection = (__bridge NSString *)(isRTL ? kCUIUserInterfaceLayoutDirectionRightToLeft : kCUIUserInterfaceLayoutDirectionLeftToRight);

    CGContextStateSaver stateSaver(cgContext);

    [[NSAppearance currentDrawingAppearance] _drawInRect:drawingTrackRect context:cgContext options:@{
        (__bridge NSString *)kCUIWidgetKey: (__bridge NSString *)kCUIWidgetSwitchFillMask,
        (__bridge NSString *)kCUISizeKey: coreUISize,
        (__bridge NSString *)kCUIUserInterfaceLayoutDirectionKey: coreUIDirection,
        (__bridge NSString *)kCUIScaleKey: @(deviceScaleFactor),
    }];

    return maskImage;
}

} // namespace WebCore::SwitchMacUtilities

#endif // PLATFORM(MAC)
