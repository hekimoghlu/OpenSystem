/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
#include "config.h"
#include "SliderTrackAdwaita.h"

#include "GraphicsContextStateSaver.h"

#if USE(THEME_ADWAITA)

namespace WebCore {
using namespace WebCore::Adwaita;

SliderTrackAdwaita::SliderTrackAdwaita(ControlPart& part, ControlFactoryAdwaita& controlFactory)
    : ControlAdwaita(part, controlFactory)
{
}

void SliderTrackAdwaita::draw(GraphicsContext& graphicsContext, const FloatRoundedRect& borderRect, float /*deviceScaleFactor*/, const ControlStyle& style)
{
    auto& sliderTrackPart = owningSliderTrackPart();
    GraphicsContextStateSaver stateSaver(graphicsContext);

    FloatRect rect = borderRect.rect();
    FloatRect fieldRect = rect;
    bool isHorizontal = sliderTrackPart.type() == StyleAppearance::SliderHorizontal;
    if (isHorizontal) {
        fieldRect.move(0, rect.height() / 2 - (sliderTrackSize / 2));
        fieldRect.setHeight(sliderTrackSize);
    } else {
        fieldRect.move(rect.width() / 2 - (sliderTrackSize / 2), 0);
        fieldRect.setWidth(sliderTrackSize);
    }

    SRGBA<uint8_t> sliderTrackBackgroundColor;

    if (style.states.contains(ControlStyle::State::DarkAppearance))
        sliderTrackBackgroundColor = sliderTrackBackgroundColorDark;
    else
        sliderTrackBackgroundColor = sliderTrackBackgroundColorLight;

    if (!style.states.contains(ControlStyle::State::Enabled))
        graphicsContext.beginTransparencyLayer(disabledOpacity);

    FloatSize corner(3, 3);
    Path path;

    path.addRoundedRect(fieldRect, corner);
    graphicsContext.setFillRule(WindRule::NonZero);
    graphicsContext.setFillColor(sliderTrackBackgroundColor);
    graphicsContext.fillPath(path);
    path.clear();

    FloatRect rangeRect = fieldRect;
    FloatRoundedRect::Radii corners;
    if (isHorizontal) {
        float offset = rangeRect.width() * sliderTrackPart.thumbPosition();
        if (style.states.contains(ControlStyle::State::InlineFlippedWritingMode)) {
            rangeRect.move(rangeRect.width() - offset, 0);
            rangeRect.setWidth(offset);
            corners.setTopRight(corner);
            corners.setBottomRight(corner);
        } else {
            rangeRect.setWidth(offset);
            corners.setTopLeft(corner);
            corners.setBottomLeft(corner);
        }
    } else {
        float offset = rangeRect.height() * sliderTrackPart.thumbPosition();
        if (style.states.contains(ControlStyle::State::VerticalWritingMode)) {
            rangeRect.setHeight(offset);
            corners.setTopLeft(corner);
            corners.setTopRight(corner);
        } else {
            rangeRect.move(0, rangeRect.height() - offset);
            rangeRect.setHeight(offset);
            corners.setBottomLeft(corner);
            corners.setBottomRight(corner);
        }
    }

    path.addRoundedRect(FloatRoundedRect(rangeRect, corners));
    graphicsContext.setFillRule(WindRule::NonZero);
    graphicsContext.setFillColor(accentColor(style));
    graphicsContext.fillPath(path);

    sliderTrackPart.drawTicks(graphicsContext, borderRect.rect(), style);

    if (style.states.contains(ControlStyle::State::Focused)) {
        // Sliders support accent-color, so we want to color their focus rings too
        Color focusRingColor = accentColor(style).colorWithAlphaMultipliedBy(focusRingOpacity);
        Adwaita::paintFocus(graphicsContext, fieldRect, sliderTrackFocusOffset, focusRingColor, Adwaita::PaintRounded::Yes);
    }

    if (!style.states.contains(ControlStyle::State::Enabled))
        graphicsContext.endTransparencyLayer();
}

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
