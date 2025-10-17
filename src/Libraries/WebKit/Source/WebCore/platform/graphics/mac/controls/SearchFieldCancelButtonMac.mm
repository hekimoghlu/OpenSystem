/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#import "SearchFieldCancelButtonMac.h"

#if PLATFORM(MAC)

#import "ControlFactoryMac.h"
#import "FloatRoundedRect.h"
#import "GraphicsContext.h"
#import "LocalDefaultSystemAppearance.h"
#import "SearchFieldCancelButtonPart.h"

namespace WebCore {

SearchFieldCancelButtonMac::SearchFieldCancelButtonMac(SearchFieldCancelButtonPart& owningPart, ControlFactoryMac& controlFactory, NSSearchFieldCell *searchFieldCell)
    : SearchControlMac(owningPart, controlFactory, searchFieldCell)
{
    ASSERT(searchFieldCell);
}

IntSize SearchFieldCancelButtonMac::cellSize(NSControlSize controlSize, const ControlStyle&) const
{
    static constexpr std::array sizes {
        IntSize { 22, 22 },
        IntSize { 19, 19 },
        IntSize { 15, 15 },
        IntSize { 22, 22 }
    };
    return sizes[controlSize];
}

FloatRect SearchFieldCancelButtonMac::rectForBounds(const FloatRect& bounds, const ControlStyle& style) const
{
    auto sizeBasedOnFontSize = sizeForSystemFont(style);
    auto diff = bounds.size() - FloatSize(sizeBasedOnFontSize);
    if (diff.isZero())
        return bounds;

    // Vertically centered and right aligned.
    auto location = bounds.location() + FloatSize { diff.width(), diff.height() / 2 };
    return { location, sizeBasedOnFontSize };
}

void SearchFieldCancelButtonMac::updateCellStates(const FloatRect& rect, const ControlStyle& style)
{
    bool enabled = style.states.contains(ControlStyle::State::Enabled);
    bool readOnly = style.states.contains(ControlStyle::State::ReadOnly);

    if (!enabled && !readOnly)
        updatePressedState([m_searchFieldCell cancelButtonCell], style);
    else if ([[m_searchFieldCell cancelButtonCell] isHighlighted])
        [[m_searchFieldCell cancelButtonCell] setHighlighted:NO];

    SearchControlMac::updateCellStates(rect, style);
}

void SearchFieldCancelButtonMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    GraphicsContextStateSaver stateSaver(context);

    auto logicalRect = rectForBounds(borderRect.rect(), style);
    if (style.zoomFactor != 1) {
        logicalRect.scale(1 / style.zoomFactor);
        context.scale(style.zoomFactor);
    }

    // Never draw a focus ring for the cancel button.
    auto styleForDrawing = style;
    styleForDrawing.states.remove(ControlStyle::State::Focused);

    drawCell(context, logicalRect, deviceScaleFactor, styleForDrawing, [m_searchFieldCell cancelButtonCell], true);
}

} // namespace WebCore

#endif // PLATFORM(MAC)
