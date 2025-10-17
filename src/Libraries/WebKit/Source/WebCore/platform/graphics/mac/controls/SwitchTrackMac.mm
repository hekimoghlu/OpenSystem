/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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
#import "SwitchTrackMac.h"

#if PLATFORM(MAC)

#import "ColorCocoa.h"
#import "ControlFactoryMac.h"
#import "FloatRoundedRect.h"
#import "GraphicsContext.h"
#import "ImageBuffer.h"
#import "LocalCurrentGraphicsContext.h"
#import "LocalDefaultSystemAppearance.h"
#import "SwitchMacUtilities.h"
#import <pal/spi/mac/CoreUISPI.h>
#import <pal/spi/mac/NSAppearanceSPI.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SwitchTrackMac);

SwitchTrackMac::SwitchTrackMac(SwitchTrackPart& part, ControlFactoryMac& controlFactory)
    : ControlMac(part, controlFactory)
{
    ASSERT(part.type() == StyleAppearance::SwitchTrack);
}

IntSize SwitchTrackMac::cellSize(NSControlSize controlSize, const ControlStyle&) const
{
    return SwitchMacUtilities::cellSize(controlSize);
}

IntOutsets SwitchTrackMac::cellOutsets(NSControlSize controlSize, const ControlStyle&) const
{
    return SwitchMacUtilities::cellOutsets(controlSize);
}

FloatRect SwitchTrackMac::rectForBounds(const FloatRect& bounds, const ControlStyle&) const
{
    return SwitchMacUtilities::rectForBounds(bounds);
}

static RefPtr<ImageBuffer> trackImage(GraphicsContext& context, RefPtr<ImageBuffer> trackMaskImage, FloatSize trackRectSize, float deviceScaleFactor, const ControlStyle& style, bool isOn, bool isInlineFlipped, bool isVertical, bool isEnabled, bool isPressed, bool isInActiveWindow, bool needsOnOffLabels, NSString *coreUISize)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    auto drawingTrackRect = NSMakeRect(0, 0, trackRectSize.width(), trackRectSize.height());

    auto trackImage = context.createImageBuffer(trackRectSize, deviceScaleFactor);

    if (!trackImage)
        return nullptr;

    auto cgContext = trackImage->context().platformContext();

    auto coreUIValue = @(isOn ? 1 : 0);
    auto coreUIState = (__bridge NSString *)(!isEnabled ? kCUIStateDisabled : isPressed ? kCUIStatePressed : kCUIStateActive);
    auto coreUIPresentation = (__bridge NSString *)(isInActiveWindow ? kCUIPresentationStateActiveKey : kCUIPresentationStateInactive);
    auto coreUIDirection = (__bridge NSString *)(isInlineFlipped ? kCUIUserInterfaceLayoutDirectionRightToLeft : kCUIUserInterfaceLayoutDirectionLeftToRight);

    CGContextStateSaver stateSaver(cgContext);

    // FIXME: clipping in context() might not always be accurate for context().platformContext().
    trackImage->context().clipToImageBuffer(*trackMaskImage, drawingTrackRect);

    [[NSAppearance currentDrawingAppearance] _drawInRect:drawingTrackRect context:cgContext options:@{
        (__bridge NSString *)kCUIWidgetKey: (__bridge NSString *)kCUIWidgetSwitchFill,
        (__bridge NSString *)kCUIStateKey: coreUIState,
        (__bridge NSString *)kCUIValueKey: coreUIValue,
        (__bridge NSString *)kCUIPresentationStateKey: coreUIPresentation,
        (__bridge NSString *)kCUISizeKey: coreUISize,
        (__bridge NSString *)kCUIUserInterfaceLayoutDirectionKey: coreUIDirection,
        (__bridge NSString *)kCUIScaleKey: @(deviceScaleFactor),
    }];

    [[NSAppearance currentDrawingAppearance] _drawInRect:drawingTrackRect context:cgContext options:@{
        (__bridge NSString *)kCUIWidgetKey: (__bridge NSString *)kCUIWidgetSwitchBorder,
        (__bridge NSString *)kCUISizeKey: coreUISize,
        (__bridge NSString *)kCUIUserInterfaceLayoutDirectionKey: coreUIDirection,
        (__bridge NSString *)kCUIScaleKey: @(deviceScaleFactor),
    }];

    if (needsOnOffLabels) {
        // This ensures the on label continues to appear upright.
        if (isVertical && isOn) {
            auto isRegularSize = coreUISize == (__bridge NSString *)kCUISizeRegular;
            if (isInlineFlipped) {
                auto thumbLogicalLeftAxis = trackRectSize.width() - trackRectSize.height();
                auto y = -thumbLogicalLeftAxis;
                trackImage->context().translate(thumbLogicalLeftAxis, y);
            }
            if (!isInlineFlipped && isRegularSize)
                trackImage->context().translate(0.0f, 1.f);
            SwitchMacUtilities::rotateContextForVerticalWritingMode(trackImage->context(), drawingTrackRect);
        }

        [[NSAppearance currentDrawingAppearance] _drawInRect:drawingTrackRect context:cgContext options:@{
            (__bridge NSString *)kCUIWidgetKey: (__bridge NSString *)kCUIWidgetSwitchOnOffLabel,
            // FIXME: Below does not pass kCUIStatePressed like NSCoreUIStateForSwitchState does,
            // as passing that does not appear to work correctly. Might be related to
            // rdar://118563716.
            (__bridge NSString *)kCUIStateKey: (__bridge NSString *)(!isEnabled ? kCUIStateDisabled : kCUIStateActive),
            (__bridge NSString *)kCUIValueKey: coreUIValue,
            (__bridge NSString *)kCUIPresentationStateKey: coreUIPresentation,
            (__bridge NSString *)kCUISizeKey: coreUISize,
            (__bridge NSString *)kCUIUserInterfaceLayoutDirectionKey: coreUIDirection,
            (__bridge NSString *)kCUIScaleKey: @(deviceScaleFactor),
        }];
    }

    return trackImage;
}

void SwitchTrackMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    GraphicsContextStateSaver stateSaver(context);

    auto isOn = owningPart().isOn();
    auto isInlineFlipped = style.states.contains(ControlStyle::State::InlineFlippedWritingMode);
    auto isVertical = style.states.contains(ControlStyle::State::VerticalWritingMode);
    auto isEnabled = style.states.contains(ControlStyle::State::Enabled);
    auto isPressed = style.states.contains(ControlStyle::State::Pressed);
    auto isInActiveWindow = style.states.contains(ControlStyle::State::WindowActive);
    auto isFocused = style.states.contains(ControlStyle::State::Focused);
    auto needsOnOffLabels = userPrefersWithoutColorDifferentiation();
    auto progress = SwitchMacUtilities::easeInOut(owningPart().progress());

    auto logicalBounds = SwitchMacUtilities::rectWithTransposedSize(borderRect.rect(), isVertical);
    auto controlSize = controlSizeForSize(logicalBounds.size(), style);
    auto size = SwitchMacUtilities::visualCellSize(cellSize(controlSize, style), style);
    auto outsets = SwitchMacUtilities::visualCellOutsets(controlSize, isVertical);

    auto trackRect = SwitchMacUtilities::trackRectForBounds(logicalBounds, size);
    auto inflatedTrackRect = inflatedRect(trackRect, size, outsets, style);
    if (isVertical)
        inflatedTrackRect.setSize(inflatedTrackRect.size().transposedSize());

    if (style.zoomFactor != 1) {
        inflatedTrackRect.scale(1 / style.zoomFactor);
        trackRect.scale(1 / style.zoomFactor);
        context.scale(style.zoomFactor);
    }

    auto coreUISize = SwitchMacUtilities::coreUISizeForControlSize(controlSize);

    auto maskImage = SwitchMacUtilities::trackMaskImage(context, inflatedTrackRect.size(), deviceScaleFactor, isInlineFlipped, coreUISize);
    if (!maskImage)
        return;

    auto createTrackImage = [&](bool isOn) {
        return trackImage(context, maskImage, inflatedTrackRect.size(), deviceScaleFactor, style, isOn, isInlineFlipped, isVertical, isEnabled, isPressed, isInActiveWindow, needsOnOffLabels, coreUISize);
    };

    RefPtr<ImageBuffer> trackImage;
    if (progress == 0.0f || progress == 1.0f) {
        trackImage = createTrackImage(progress == 0.0f ? !isOn : isOn);
        if (!trackImage)
            return;
    } else {
        auto fromImage = createTrackImage(!isOn);
        if (!fromImage)
            return;
        auto toImage = createTrackImage(isOn);
        if (!toImage)
            return;
        trackImage = context.createImageBuffer(inflatedTrackRect.size(), deviceScaleFactor);
        if (!trackImage)
            return;
        // This logic is from CrossfadeGeneratedImage.h, but we copy it to avoid some overhead and
        // also because that class is not supposed to be used in GPUP.
        // FIXME: As above, not using context().platformContext() here is likely dubious.
        trackImage->context().setAlpha(1.0f - progress);
        trackImage->context().drawConsumingImageBuffer(WTFMove(fromImage), IntPoint(), ImagePaintingOptions { CompositeOperator::SourceOver });
        trackImage->context().setAlpha(progress);
        trackImage->context().drawConsumingImageBuffer(WTFMove(toImage), IntPoint(), ImagePaintingOptions { CompositeOperator::PlusLighter });
    }

    {
        GraphicsContextStateSaver rotationStateSaver(context);
        if (isVertical)
            SwitchMacUtilities::rotateContextForVerticalWritingMode(context, inflatedTrackRect);
        context.drawConsumingImageBuffer(WTFMove(trackImage), inflatedTrackRect.location());
    }

    if (isFocused) {
        auto color = colorFromCocoaColor([NSColor keyboardFocusIndicatorColor]).opaqueColor();
        context.drawFocusRing(Vector { trackRect }, 0, 0, color);
    }
}

} // namespace WebCore

#endif // PLATFORM(MAC)
