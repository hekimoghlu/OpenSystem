/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 16, 2025.
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
#import "ToggleButtonMac.h"

#if PLATFORM(MAC)

#import "ControlFactoryMac.h"
#import "FloatRoundedRect.h"
#import "GraphicsContext.h"
#import "LocalDefaultSystemAppearance.h"
#import "ToggleButtonPart.h"
#import <pal/spi/cocoa/NSButtonCellSPI.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ToggleButtonMac);

ToggleButtonMac::ToggleButtonMac(ToggleButtonPart& owningPart, ControlFactoryMac& controlFactory, NSButtonCell *buttonCell)
    : ButtonControlMac(owningPart, controlFactory, buttonCell)
{
    ASSERT(m_owningPart.type() == StyleAppearance::Checkbox || m_owningPart.type() == StyleAppearance::Radio);
}

IntSize ToggleButtonMac::cellSize(NSControlSize controlSize, const ControlStyle& style) const
{
    static const std::array<IntSize, 4> checkboxSizes =
    {
        IntSize { 14, 14 },
        IntSize { 12, 12 },
        IntSize { 10, 10 },
        IntSize { 16, 16 }
    };
    static const std::array<IntSize, 4> radioSizes =
    {
        IntSize { 16, 16 },
        IntSize { 12, 12 },
        IntSize { 10, 10 },
        IntSize { 0, 0 }
    };
    static const std::array<IntSize, 4> largeRadioSizes =
    {
        IntSize { 14, 14 },
        IntSize { 12, 12 },
        IntSize { 10, 10 },
        IntSize { 16, 16 }
    };

    if (m_owningPart.type() == StyleAppearance::Checkbox)
        return checkboxSizes[controlSize];

    if (style.states.contains(ControlStyle::State::LargeControls))
        return largeRadioSizes[controlSize];

    return radioSizes[controlSize];
}

IntOutsets ToggleButtonMac::cellOutsets(NSControlSize controlSize, const ControlStyle&) const
{
    static const std::array checkboxOutsets {
        // top right bottom left
        IntOutsets { 2, 2, 2, 2 },
        IntOutsets { 2, 1, 2, 1 },
        IntOutsets { 0, 0, 1, 0 },
        IntOutsets { 2, 2, 2, 2 },
    };
    static const std::array radioOutsets {
        // top right bottom left
        IntOutsets { 1, 0, 1, 2 },
        IntOutsets { 1, 1, 2, 1 },
        IntOutsets { 0, 0, 1, 1 },
        IntOutsets { 1, 0, 1, 2 },
    };
    return (m_owningPart.type() == StyleAppearance::Checkbox ? checkboxOutsets : radioOutsets)[controlSize];
}

FloatRect ToggleButtonMac::rectForBounds(const FloatRect& bounds, const ControlStyle& style) const
{
    auto controlSize = [m_buttonCell controlSize];

    FloatSize size = cellSize(controlSize, style);
    size.scale(style.zoomFactor);

    auto outsets = cellOutsets(controlSize, style);

    return inflatedRect(bounds, size, outsets, style);
}

void ToggleButtonMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    GraphicsContextStateSaver stateSaver(context);

    auto logicalRect = rectForBounds(borderRect.rect(), style);

    if (style.zoomFactor != 1) {
        logicalRect.scale(1 / style.zoomFactor);
        context.scale(style.zoomFactor);
    }

    if ([m_buttonCell _stateAnimationRunning]) {
        context.translate(logicalRect.location());
        context.scale(FloatSize(1, -1));
        context.translate(0, -logicalRect.height());

        [m_buttonCell _renderCurrentAnimationFrameInContext:context.platformContext() atLocation:NSMakePoint(0, 0)];

        if (![m_buttonCell _stateAnimationRunning] && style.states.contains(ControlStyle::State::Focused))
            drawCell(context, logicalRect, deviceScaleFactor, style, m_buttonCell.get(), false);
    } else
        drawCell(context, logicalRect, deviceScaleFactor, style, m_buttonCell.get(), true);
}

} // namespace WebCore

#endif // PLATFORM(MAC)
