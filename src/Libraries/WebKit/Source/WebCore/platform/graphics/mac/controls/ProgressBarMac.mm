/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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
#import "ProgressBarMac.h"

#if PLATFORM(MAC)

#import "GraphicsContext.h"
#import "ImageBuffer.h"
#import "LocalDefaultSystemAppearance.h"
#import "ProgressBarPart.h"
#import <pal/spi/mac/CoreUISPI.h>
#import <pal/spi/mac/NSAppearanceSPI.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ProgressBarMac);

ProgressBarMac::ProgressBarMac(ProgressBarPart& owningPart, ControlFactoryMac& controlFactory)
    : ControlMac(owningPart, controlFactory)
{
}

IntSize ProgressBarMac::cellSize(NSControlSize controlSize, const ControlStyle&) const
{
    static const std::array<IntSize, 4> sizes =
    {
        IntSize(0, 20),
        IntSize(0, 12),
        IntSize(0, 12),
        IntSize(0, 20)
    };
    return sizes[controlSize];
}

IntOutsets ProgressBarMac::cellOutsets(NSControlSize controlSize, const ControlStyle&) const
{
    static const std::array cellOutsets {
        // top right bottom left
        IntOutsets { 0, 0, 1, 0 },
        IntOutsets { 0, 0, 1, 0 },
        IntOutsets { 0, 0, 1, 0 },
        IntOutsets { 0, 0, 1, 0 },
    };
    return cellOutsets[controlSize];
}

FloatRect ProgressBarMac::rectForBounds(const FloatRect& bounds, const ControlStyle& style) const
{
    auto isVerticalWritingMode = style.states.contains(ControlStyle::State::VerticalWritingMode);

    auto logicalBounds = bounds;
    if (isVerticalWritingMode)
        logicalBounds.setSize(bounds.size().transposedSize());

    int minimumProgressBarBlockSize = sizeForSystemFont(style).height();
    if (logicalBounds.height() > minimumProgressBarBlockSize)
        return bounds;

    auto controlSize = controlSizeForFont(style);
    auto size = cellSize(controlSize, style);
    auto outsets = cellOutsets(controlSize, style);

    size.scale(style.zoomFactor);

    if (isVerticalWritingMode) {
        outsets = { outsets.left(), outsets.top(), outsets.right(), outsets.bottom() };
        size.setWidth(size.height());
        size.setHeight(bounds.height());
    } else
        size.setWidth(bounds.width());

    // Make enough room for the shadow.
    return inflatedRect(bounds, size, outsets, style);
}

void ProgressBarMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    auto isVerticalWritingMode = style.states.contains(ControlStyle::State::VerticalWritingMode);

    auto inflatedRect = rectForBounds(borderRect.rect(), style);
    if (isVerticalWritingMode)
        inflatedRect.setSize(inflatedRect.size().transposedSize());

    auto imageBuffer = context.createImageBuffer(inflatedRect.size(), deviceScaleFactor);
    if (!imageBuffer)
        return;

    CGContextRef cgContext = imageBuffer->context().platformContext();

    auto& progressBarPart = owningProgressBarPart();
    auto controlSize = controlSizeForFont(style);
    bool isIndeterminate = progressBarPart.position() < 0;
    bool isActive = style.states.contains(ControlStyle::State::WindowActive);

    auto coreUISizeForProgressBarSize = [](NSControlSize size) -> CFStringRef {
        switch (size) {
        case NSControlSizeMini:
        case NSControlSizeSmall:
            return kCUISizeSmall;
        case NSControlSizeRegular:
        case NSControlSizeLarge:
            return kCUISizeRegular;
        }
        ASSERT_NOT_REACHED();
        return nullptr;
    };

    [[NSAppearance currentDrawingAppearance] _drawInRect:NSMakeRect(0, 0, inflatedRect.width(), inflatedRect.height()) context:cgContext options:@{
        (__bridge NSString *)kCUIWidgetKey: (__bridge NSString *)(isIndeterminate ? kCUIWidgetProgressIndeterminateBar : kCUIWidgetProgressBar),
        (__bridge NSString *)kCUIValueKey: @(isIndeterminate ? 1 : std::min(nextafter(1.0, -1), progressBarPart.position())),
        (__bridge NSString *)kCUISizeKey: (__bridge NSString *)coreUISizeForProgressBarSize(controlSize),
        (__bridge NSString *)kCUIUserInterfaceLayoutDirectionKey: (__bridge NSString *)kCUIUserInterfaceLayoutDirectionLeftToRight,
        (__bridge NSString *)kCUIScaleKey: @(deviceScaleFactor),
        (__bridge NSString *)kCUIPresentationStateKey: (__bridge NSString *)(isActive ? kCUIPresentationStateActiveKey : kCUIPresentationStateInactive),
        (__bridge NSString *)kCUIOrientationKey: (__bridge NSString *)kCUIOrientHorizontal,
        (__bridge NSString *)kCUIAnimationStartTimeKey: @(progressBarPart.animationStartTime().seconds()),
        (__bridge NSString *)kCUIAnimationTimeKey: @(MonotonicTime::now().secondsSinceEpoch().seconds())
    }];

    GraphicsContextStateSaver stateSaver(context);

    if (isVerticalWritingMode) {
        context.translate(inflatedRect.height(), 0);
        context.translate(inflatedRect.location());
        context.rotate(piOverTwoFloat);
        context.translate(-inflatedRect.location());
    }

    if (style.states.contains(ControlStyle::State::InlineFlippedWritingMode)) {
        context.translate(2 * inflatedRect.x() + inflatedRect.width(), 0);
        context.scale(FloatSize(-1, 1));
    }

    context.drawConsumingImageBuffer(WTFMove(imageBuffer), inflatedRect.location());
}

} // namespace WebCore

#endif // PLATFORM(MAC)
