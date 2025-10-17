/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
#import "SwitchThumbMac.h"

#if PLATFORM(MAC)

#import "ControlFactoryMac.h"
#import "FloatRoundedRect.h"
#import "GraphicsContext.h"
#import "LocalCurrentGraphicsContext.h"
#import "LocalDefaultSystemAppearance.h"
#import "SwitchMacUtilities.h"
#import <pal/spi/mac/CoreUISPI.h>
#import <pal/spi/mac/NSAppearanceSPI.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SwitchThumbMac);

SwitchThumbMac::SwitchThumbMac(SwitchThumbPart& part, ControlFactoryMac& controlFactory)
    : ControlMac(part, controlFactory)
{
    ASSERT(part.type() == StyleAppearance::SwitchThumb);
}

IntSize SwitchThumbMac::cellSize(NSControlSize controlSize, const ControlStyle&) const
{
    return SwitchMacUtilities::cellSize(controlSize);
}

IntOutsets SwitchThumbMac::cellOutsets(NSControlSize controlSize, const ControlStyle&) const
{
    return SwitchMacUtilities::cellOutsets(controlSize);
}

FloatRect SwitchThumbMac::rectForBounds(const FloatRect& bounds, const ControlStyle&) const
{
    return SwitchMacUtilities::rectForBounds(bounds);
}

void SwitchThumbMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    GraphicsContextStateSaver stateSaver(context);

    auto isOn = owningPart().isOn();
    auto isInlineFlipped = style.states.contains(ControlStyle::State::InlineFlippedWritingMode);
    auto isVertical = style.states.contains(ControlStyle::State::VerticalWritingMode);
    auto isEnabled = style.states.contains(ControlStyle::State::Enabled);
    auto isPressed = style.states.contains(ControlStyle::State::Pressed);
    auto progress = SwitchMacUtilities::easeInOut(owningPart().progress());

    auto logicalBounds = SwitchMacUtilities::rectWithTransposedSize(borderRect.rect(), isVertical);
    auto controlSize = controlSizeForSize(logicalBounds.size(), style);
    auto logicalTrackSize = cellSize(controlSize, style);
    auto logicalThumbSize = IntSize { logicalTrackSize.height(), logicalTrackSize.height() };
    auto trackSize = SwitchMacUtilities::visualCellSize(logicalTrackSize, style);
    auto thumbSize = SwitchMacUtilities::visualCellSize(logicalThumbSize, style);
    auto outsets = SwitchMacUtilities::visualCellOutsets(controlSize, isVertical);

    auto trackRect = SwitchMacUtilities::trackRectForBounds(logicalBounds, trackSize);
    auto thumbRect = SwitchMacUtilities::trackRectForBounds(logicalBounds, thumbSize);

    auto inflatedTrackRect = inflatedRect(trackRect, trackSize, outsets, style);
    auto inflatedThumbRect = inflatedRect(thumbRect, thumbSize, outsets, style);
    if (isVertical) {
        inflatedTrackRect.setSize(inflatedTrackRect.size().transposedSize());
        inflatedThumbRect.setSize(inflatedThumbRect.size().transposedSize());
    }

    if (style.zoomFactor != 1) {
        inflatedTrackRect.scale(1 / style.zoomFactor);
        inflatedThumbRect.scale(1 / style.zoomFactor);
        context.scale(style.zoomFactor);
    }

    auto drawingThumbIsLogicallyLeft = (!isInlineFlipped && !isOn) || (isInlineFlipped && isOn);
    auto drawingThumbLogicalXAxis = inflatedTrackRect.width() - inflatedThumbRect.width();
    auto drawingThumbLogicalXAxisProgress = drawingThumbLogicalXAxis * progress;
    auto drawingThumbLogicalX = drawingThumbIsLogicallyLeft ? drawingThumbLogicalXAxis - drawingThumbLogicalXAxisProgress : drawingThumbLogicalXAxisProgress;
    auto drawingThumbRect = NSMakeRect(drawingThumbLogicalX, 0, inflatedThumbRect.width(), inflatedThumbRect.height());

    auto coreUISize = SwitchMacUtilities::coreUISizeForControlSize(controlSize);

    auto maskImage = SwitchMacUtilities::trackMaskImage(context, inflatedTrackRect.size(), deviceScaleFactor, isInlineFlipped, coreUISize);
    if (!maskImage)
        return;

    auto trackImage = context.createImageBuffer(inflatedTrackRect.size(), deviceScaleFactor);
    if (!trackImage)
        return;

    auto cgContext = trackImage->context().platformContext();

    {
        CGContextStateSaver stateSaverTrack(cgContext);

        // FIXME: clipping in context() might not always be accurate for context().platformContext().
        trackImage->context().clipToImageBuffer(*maskImage, NSMakeRect(0, 0, inflatedTrackRect.width(), inflatedTrackRect.height()));

        [[NSAppearance currentDrawingAppearance] _drawInRect:drawingThumbRect context:cgContext options:@{
            (__bridge NSString *)kCUIWidgetKey: (__bridge NSString *)kCUIWidgetSwitchKnob,
            (__bridge NSString *)kCUIStateKey: (__bridge NSString *)(!isEnabled ? kCUIStateDisabled : isPressed ? kCUIStatePressed : kCUIStateActive),
            (__bridge NSString *)kCUISizeKey: SwitchMacUtilities::coreUISizeForControlSize(controlSize),
            (__bridge NSString *)kCUIUserInterfaceLayoutDirectionKey: (__bridge NSString *)(isInlineFlipped ? kCUIUserInterfaceLayoutDirectionRightToLeft : kCUIUserInterfaceLayoutDirectionLeftToRight),
            (__bridge NSString *)kCUIScaleKey: @(deviceScaleFactor),
        }];
    }

    if (isVertical)
        SwitchMacUtilities::rotateContextForVerticalWritingMode(context, inflatedTrackRect);

    context.drawConsumingImageBuffer(WTFMove(trackImage), inflatedTrackRect.location());
}

} // namespace WebCore

#endif // PLATFORM(MAC)
