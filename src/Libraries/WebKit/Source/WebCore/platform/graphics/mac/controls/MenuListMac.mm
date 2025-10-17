/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
#import "MenuListMac.h"

#if PLATFORM(MAC)

#import "ControlFactoryMac.h"
#import "GraphicsContext.h"
#import "LocalDefaultSystemAppearance.h"
#import "MenuListPart.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MenuListMac);

MenuListMac::MenuListMac(MenuListPart& owningPart, ControlFactoryMac& controlFactory, NSPopUpButtonCell *popUpButtonCell)
    : ControlMac(owningPart, controlFactory)
    , m_popUpButtonCell(popUpButtonCell)
{
    ASSERT(m_popUpButtonCell);
}

IntSize MenuListMac::cellSize(NSControlSize controlSize, const ControlStyle&) const
{
    static constexpr std::array sizes {
        IntSize { 0, 21 },
        IntSize { 0, 18 },
        IntSize { 0, 15 },
        IntSize { 0, 24 }
    };
    return sizes[controlSize];
}

IntOutsets MenuListMac::cellOutsets(NSControlSize controlSize, const ControlStyle&) const
{
    static const std::array outsets {
        // top right bottom left
        IntOutsets { 0, 3, 1, 3 },
        IntOutsets { 0, 3, 2, 3 },
        IntOutsets { 0, 1, 0, 1 },
        IntOutsets { 0, 6, 2, 6 },
    };
    return outsets[controlSize];
}

void MenuListMac::updateCellStates(const FloatRect& rect, const ControlStyle& style)
{
    ControlMac::updateCellStates(rect, style);

    auto direction = style.states.contains(ControlStyle::State::InlineFlippedWritingMode) ? NSUserInterfaceLayoutDirectionRightToLeft : NSUserInterfaceLayoutDirectionLeftToRight;
    [m_popUpButtonCell setUserInterfaceLayoutDirection:direction];

    // Update the various states we respond to.
    updateCheckedState(m_popUpButtonCell.get(), style);
    updateEnabledState(m_popUpButtonCell.get(), style);
    updatePressedState(m_popUpButtonCell.get(), style);

    // Only update if we have to, since AppKit does work even if the size is the same.
    auto controlSize = controlSizeForSize(rect.size(), style);
    if (controlSize != [m_popUpButtonCell controlSize])
        [m_popUpButtonCell setControlSize:controlSize];
}

FloatRect MenuListMac::rectForBounds(const FloatRect& bounds, const ControlStyle& style) const
{
    int minimumMenuListSize = sizeForSystemFont(style).width();
    if (bounds.width() < minimumMenuListSize)
        return bounds;

    auto controlSize = [m_popUpButtonCell controlSize];
    auto size = cellSize(controlSize, style);
    auto outsets = cellOutsets(controlSize, style);

    size.scale(style.zoomFactor);
    size.setWidth(bounds.width());

    // Make enough room for the shadow.
    return inflatedRect(bounds, size, outsets, style);
}

void MenuListMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    GraphicsContextStateSaver stateSaver(context);

    auto inflatedRect = rectForBounds(borderRect.rect(), style);

    if (style.zoomFactor != 1) {
        inflatedRect.scale(1 / style.zoomFactor);
        context.scale(style.zoomFactor);
    }

    drawCell(context, inflatedRect, deviceScaleFactor, style, m_popUpButtonCell.get(), true);
}

} // namespace WebCore

#endif // PLATFORM(MAC)
