/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
#import "InnerSpinButtonMac.h"

#if PLATFORM(MAC)

#import "FloatRoundedRect.h"
#import "GraphicsContext.h"
#import "InnerSpinButtonPart.h"
#import "LocalDefaultSystemAppearance.h"
#import <pal/spi/cocoa/NSStepperCellSPI.h>
#import <pal/spi/mac/CoreUISPI.h>
#import <pal/spi/mac/NSAppearanceSPI.h>
#import <wtf/BlockObjCExceptions.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(InnerSpinButtonMac);

InnerSpinButtonMac::InnerSpinButtonMac(InnerSpinButtonPart& owningPart, ControlFactoryMac& controlFactory, NSStepperCell *stepperCell)
    : ControlMac(owningPart, controlFactory)
    , m_stepperCell(stepperCell)
{
}

#if HAVE(NSSTEPPERCELL_INCREMENTING)
IntOutsets InnerSpinButtonMac::cellOutsets(NSControlSize controlSize, const ControlStyle&) const
{
    static const std::array cellOutsets {
        // top right bottom left
        IntOutsets { 4, 3, 4, 3 },
        IntOutsets { 2, 2, 4, 2 },
        IntOutsets { 2, 2, 3, 2 },
        IntOutsets { 4, 3, 4, 3 },
    };
    return cellOutsets[controlSize];
}
#endif

IntSize InnerSpinButtonMac::cellSize(NSControlSize controlSize, const ControlStyle&) const
{
#if HAVE(NSSTEPPERCELL_INCREMENTING)
    if ([NSStepperCell instancesRespondToSelector:@selector(setIncrementing:)]) {
        static constexpr std::array sizes {
            IntSize { 19, 28 },
            IntSize { 15, 22 },
            IntSize { 13, 18 },
            IntSize { 19, 28 }
        };
        return sizes[controlSize];
    }
#endif
    static constexpr std::array sizes {
        IntSize { 19, 27 },
        IntSize { 15, 22 },
        IntSize { 13, 15 },
        IntSize { 19, 27 }
    };
    return sizes[controlSize];
}

void InnerSpinButtonMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
#if HAVE(NSSTEPPERCELL_INCREMENTING)
    if ([NSStepperCell instancesRespondToSelector:@selector(setIncrementing:)]) {
        drawWithCell(context, borderRect, deviceScaleFactor, style);
        return;
    }
#endif
    UNUSED_PARAM(deviceScaleFactor);
    drawWithCoreUI(context, borderRect, style);
}

#if HAVE(NSSTEPPERCELL_INCREMENTING)
void InnerSpinButtonMac::drawWithCell(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);
    GraphicsContextStateSaver stateSaver(context);

    auto logicalRect = rectForBounds(borderRect.rect(), style);

    if (style.zoomFactor != 1) {
        logicalRect.scale(1 / style.zoomFactor);
        context.scale(style.zoomFactor);
    }

    drawCell(context, logicalRect, deviceScaleFactor, style, m_stepperCell.get(), true);
}
#endif

void InnerSpinButtonMac::drawWithCoreUI(GraphicsContext& context, const FloatRoundedRect& borderRect, const ControlStyle& style)
{
    BEGIN_BLOCK_OBJC_EXCEPTIONS

    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    // We don't use NSStepperCell because there are no ways to draw an
    // NSStepperCell with the up button highlighted.

    NSString *coreUIState;
    auto states = style.states;
    if (!states.contains(ControlStyle::State::Enabled))
        coreUIState = (__bridge NSString *)kCUIStateDisabled;
    else if (states.contains(ControlStyle::State::Pressed))
        coreUIState = (__bridge NSString *)kCUIStatePressed;
    else
        coreUIState = (__bridge NSString *)kCUIStateActive;

    NSString *coreUISize;
    auto controlSize = controlSizeForSize(borderRect.rect().size(), style);
    if (controlSize == NSControlSizeMini)
        coreUISize = (__bridge NSString *)kCUISizeMini;
    else if (controlSize == NSControlSizeSmall)
        coreUISize = (__bridge NSString *)kCUISizeSmall;
    else
        coreUISize = (__bridge NSString *)kCUISizeRegular;

    IntRect logicalRect(borderRect.rect());

    GraphicsContextStateSaver stateSaver(context);
    if (style.zoomFactor != 1) {
        logicalRect.scale(1 / style.zoomFactor);
        context.scale(style.zoomFactor);
    }

    [[NSAppearance currentDrawingAppearance] _drawInRect:logicalRect context:context.platformContext() options:@{
        (__bridge NSString *)kCUIWidgetKey: (__bridge NSString *)kCUIWidgetButtonLittleArrows,
        (__bridge NSString *)kCUISizeKey: coreUISize,
        (__bridge NSString *)kCUIStateKey: coreUIState,
        (__bridge NSString *)kCUIValueKey: states.contains(ControlStyle::State::SpinUp) ? @1 : @0,
        (__bridge NSString *)kCUIIsFlippedKey: @NO,
        (__bridge NSString *)kCUIScaleKey: @1,
        (__bridge NSString *)kCUIMaskOnlyKey: @NO
    }];

    END_BLOCK_OBJC_EXCEPTIONS
}

#if HAVE(NSSTEPPERCELL_INCREMENTING)
FloatRect InnerSpinButtonMac::rectForBounds(const FloatRect& bounds, const ControlStyle& style) const
{
    auto controlSize = [m_stepperCell controlSize];

    FloatSize size = cellSize(controlSize, style);
    size.scale(style.zoomFactor);

    auto outsets = cellOutsets(controlSize, style);

    return inflatedRect(bounds, size, outsets, style);
}

void InnerSpinButtonMac::updateCellStates(const FloatRect& rect, const ControlStyle& style)
{
    ControlMac::updateCellStates(rect, style);

    if (![m_stepperCell respondsToSelector:@selector(setIncrementing:)])
        return;

    const auto& states = style.states;

    [m_stepperCell setIncrementing:states.contains(ControlStyle::State::SpinUp)];

    bool oldPressed = [m_stepperCell isHighlighted];
    bool pressed = states.contains(ControlStyle::State::Pressed);
    if (pressed != oldPressed)
        [m_stepperCell setHighlighted:pressed];

    auto controlSize = controlSizeForSize(rect.size(), style);

    if (controlSize == NSControlSizeLarge)
        controlSize = NSControlSizeRegular;

    if (controlSize != [m_stepperCell controlSize])
        [m_stepperCell setControlSize:controlSize];
}
#endif

} // namespace WebCore

#endif // PLATFORM(MAC)
