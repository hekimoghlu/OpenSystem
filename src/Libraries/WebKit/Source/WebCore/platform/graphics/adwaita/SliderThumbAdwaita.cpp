/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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
#include "SliderThumbAdwaita.h"

#include "GraphicsContextStateSaver.h"

#if USE(THEME_ADWAITA)

namespace WebCore {
using namespace WebCore::Adwaita;

SliderThumbAdwaita::SliderThumbAdwaita(ControlPart& part, ControlFactoryAdwaita& controlFactory)
    : ControlAdwaita(part, controlFactory)
{
}

void SliderThumbAdwaita::draw(GraphicsContext& graphicsContext, const FloatRoundedRect& borderRect, float /*deviceScaleFactor*/, const ControlStyle& style)
{
    GraphicsContextStateSaver stateSaver(graphicsContext);

    SRGBA<uint8_t> sliderThumbBackgroundColor;
    SRGBA<uint8_t> sliderThumbBackgroundHoveredColor;
    SRGBA<uint8_t> sliderThumbBackgroundDisabledColor;
    SRGBA<uint8_t> sliderThumbBorderColor;

    if (style.states.contains(ControlStyle::State::DarkAppearance)) {
        sliderThumbBackgroundColor = sliderThumbBackgroundColorDark;
        sliderThumbBackgroundHoveredColor = sliderThumbBackgroundHoveredColorDark;
        sliderThumbBackgroundDisabledColor = sliderThumbBackgroundDisabledColorDark;
        sliderThumbBorderColor = sliderThumbBorderColorDark;
    } else {
        sliderThumbBackgroundColor = sliderThumbBackgroundColorLight;
        sliderThumbBackgroundHoveredColor = sliderThumbBackgroundHoveredColorLight;
        sliderThumbBackgroundDisabledColor = sliderThumbBackgroundDisabledColorLight;
        sliderThumbBorderColor = sliderThumbBorderColorLight;
    }

    FloatRect fieldRect = borderRect.rect();
    Path path;
    path.addEllipseInRect(fieldRect);
    fieldRect.inflate(-sliderThumbBorderSize);
    path.addEllipseInRect(fieldRect);
    graphicsContext.setFillRule(WindRule::EvenOdd);
    if (style.states.contains(ControlStyle::State::Enabled) && style.states.contains(ControlStyle::State::Pressed))
        graphicsContext.setFillColor(accentColor(style));
    else
        graphicsContext.setFillColor(sliderThumbBorderColor);
    graphicsContext.fillPath(path);
    path.clear();

    path.addEllipseInRect(fieldRect);
    graphicsContext.setFillRule(WindRule::NonZero);
    if (!style.states.contains(ControlStyle::State::Enabled))
        graphicsContext.setFillColor(sliderThumbBackgroundDisabledColor);
    else if (style.states.contains(ControlStyle::State::Hovered))
        graphicsContext.setFillColor(sliderThumbBackgroundHoveredColor);
    else
        graphicsContext.setFillColor(sliderThumbBackgroundColor);
    graphicsContext.fillPath(path);
}

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
