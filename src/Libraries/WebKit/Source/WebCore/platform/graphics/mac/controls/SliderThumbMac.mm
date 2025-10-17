/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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
#import "SliderThumbMac.h"

#if PLATFORM(MAC)

#import "ControlFactoryMac.h"
#import "FloatRoundedRect.h"
#import "GraphicsContext.h"
#import "LocalDefaultSystemAppearance.h"
#import "SliderThumbPart.h"
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SliderThumbMac);

SliderThumbMac::SliderThumbMac(SliderThumbPart& owningPart, ControlFactoryMac& controlFactory, NSSliderCell *sliderCell)
    : ControlMac(owningPart, controlFactory)
    , m_sliderCell(sliderCell)
{
    ASSERT(m_owningPart.type() == StyleAppearance::SliderThumbHorizontal || m_owningPart.type() == StyleAppearance::SliderThumbVertical);
}

void SliderThumbMac::updateCellStates(const FloatRect& rect, const ControlStyle& style)
{
    ControlMac::updateCellStates(rect, style);

    updateEnabledState(m_sliderCell.get(), style);
    updateFocusedState(m_sliderCell.get(), style);
}

FloatRect SliderThumbMac::rectForBounds(const FloatRect& bounds, const ControlStyle& style) const
{
    if (m_owningPart.type() == StyleAppearance::SliderThumbHorizontal)
        return bounds;

    // Make the height of the vertical slider slightly larger so NSSliderCell will draw a vertical slider.
    static constexpr float verticalSliderHeightPadding = 0.1f;
    return { bounds.location(), bounds.size() + FloatSize { 0, verticalSliderHeightPadding * style.zoomFactor } };
}

void SliderThumbMac::draw(GraphicsContext& context, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    LocalDefaultSystemAppearance localAppearance(style.states.contains(ControlStyle::State::DarkAppearance), style.accentColor);

    GraphicsContextStateSaver stateSaver(context);

    auto logicalRect = rectForBounds(borderRect.rect(), style);

    if (style.zoomFactor != 1) {
        logicalRect.scale(1 / style.zoomFactor);
        context.scale(style.zoomFactor);
    }

    // Never draw a focus ring for the slider thumb.
    auto styleForDrawing = style;
    styleForDrawing.states.remove(ControlStyle::State::Focused);

    drawCell(context, logicalRect, deviceScaleFactor, styleForDrawing, m_sliderCell.get(), true);
}

} // namespace WebCore

#endif // PLATFORM(MAC)
