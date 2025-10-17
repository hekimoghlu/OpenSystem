/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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
#import "ButtonMac.h"

#if PLATFORM(MAC)

#import "ButtonPart.h"
#import "ControlFactoryMac.h"
#import "GraphicsContext.h"
#import "LocalDefaultSystemAppearance.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ButtonMac);

ButtonMac::ButtonMac(ButtonPart& owningPart, ControlFactoryMac& controlFactory, NSButtonCell *buttonCell)
    : ButtonControlMac(owningPart, controlFactory, buttonCell)
{
    ASSERT(m_owningPart.type() == StyleAppearance::Button
        || m_owningPart.type() == StyleAppearance::DefaultButton
        || m_owningPart.type() == StyleAppearance::PushButton
        || m_owningPart.type() == StyleAppearance::SquareButton);
}

IntSize ButtonMac::cellSize(NSControlSize controlSize, const ControlStyle&) const
{
    // Buttons really only constrain height. They respect width.
    static constexpr std::array cellSizes {
        IntSize { 0, 20 },
        IntSize { 0, 16 },
        IntSize { 0, 13 },
        IntSize { 0, 28 }
    };
    return cellSizes[controlSize];
}

IntOutsets ButtonMac::cellOutsets(NSControlSize controlSize, const ControlStyle&) const
{
    // FIXME: Determine these values programmatically.
    // https://bugs.webkit.org/show_bug.cgi?id=251066
    static const std::array cellOutsets {
        // top right bottom left
        IntOutsets { 5, 7, 7, 7 },
        IntOutsets { 4, 6, 7, 6 },
        IntOutsets { 1, 2, 2, 2 },
        IntOutsets { 6, 6, 6, 6 },
    };
    return cellOutsets[controlSize];
}

NSBezelStyle ButtonMac::bezelStyle(const FloatRect& rect, const ControlStyle& style) const
{
    if (m_owningPart.type() == StyleAppearance::SquareButton)
        return NSBezelStyleShadowlessSquare;

    auto controlSize = style.states.contains(ControlStyle::State::LargeControls) ? NSControlSizeLarge : NSControlSizeRegular;
    auto size = cellSize(controlSize, style);

    if (rect.height() > size.height() * style.zoomFactor)
        return NSBezelStyleShadowlessSquare;

    return NSBezelStyleRounded;
}

void ButtonMac::updateCellStates(const FloatRect& rect, const ControlStyle& style)
{
    ButtonControlMac::updateCellStates(rect, style);
    [m_buttonCell setBezelStyle:bezelStyle(rect, style)];
}

FloatRect ButtonMac::rectForBounds(const FloatRect& bounds, const ControlStyle& style) const
{
    if ([m_buttonCell bezelStyle] != NSBezelStyleRounded)
        return bounds;

    auto controlSize = [m_buttonCell controlSize];

    // Explicitly use `FloatSize` to support non-integral sizes following zoom.
    FloatSize size = cellSize(controlSize, style);
    size.scale(style.zoomFactor);
    size.setWidth(bounds.width());

    auto rect = bounds;
    auto delta = rect.height() - size.height();
    if (delta > 0)
        rect.inflateY(-delta / 2);

    auto outsets = cellOutsets(controlSize, style);
    return inflatedRect(rect, size, outsets, style);
}

void ButtonMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    GraphicsContextStateSaver stateSaver(context);

    auto inflatedRect = rectForBounds(borderRect.rect(), style);

    if (style.zoomFactor != 1) {
        inflatedRect.scale(1 / style.zoomFactor);
        context.scale(style.zoomFactor);
    }

    drawCell(context, inflatedRect, deviceScaleFactor, style, m_buttonCell.get(), true);
}

} // namespace WebCore

#endif // PLATFORM(MAC)
