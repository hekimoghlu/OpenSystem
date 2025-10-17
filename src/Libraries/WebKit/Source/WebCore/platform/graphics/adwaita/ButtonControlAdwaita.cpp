/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 21, 2023.
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
#include "ButtonControlAdwaita.h"

#include "GraphicsContextStateSaver.h"

#if USE(THEME_ADWAITA)

namespace WebCore {
using namespace WebCore::Adwaita;

ButtonControlAdwaita::ButtonControlAdwaita(ControlPart& part, ControlFactoryAdwaita& controlFactory)
    : ControlAdwaita(part, controlFactory)
{
}

void ButtonControlAdwaita::draw(GraphicsContext& graphicsContext, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    drawButton(graphicsContext, borderRect, deviceScaleFactor, style);
}

void ButtonControlAdwaita::drawButton(GraphicsContext& graphicsContext, const FloatRoundedRect& borderRect, float /*deviceScaleFactor*/, const ControlStyle& style)
{
    GraphicsContextStateSaver stateSaver(graphicsContext);

    SRGBA<uint8_t> buttonBorderColor;
    SRGBA<uint8_t> buttonBackgroundColor;
    SRGBA<uint8_t> buttonBackgroundHoveredColor;
    SRGBA<uint8_t> buttonBackgroundPressedColor;

    if (style.states.contains(ControlStyle::State::DarkAppearance)) {
        buttonBorderColor = buttonBorderColorDark;
        buttonBackgroundColor = buttonBackgroundColorDark;
        buttonBackgroundHoveredColor = buttonBackgroundHoveredColorDark;
        buttonBackgroundPressedColor = buttonBackgroundPressedColorDark;
    } else {
        buttonBorderColor = buttonBorderColorLight;
        buttonBackgroundColor = buttonBackgroundColorLight;
        buttonBackgroundHoveredColor = buttonBackgroundHoveredColorLight;
        buttonBackgroundPressedColor = buttonBackgroundPressedColorLight;
    }

    if (!style.states.contains(ControlStyle::State::Enabled))
        graphicsContext.beginTransparencyLayer(disabledOpacity);

    FloatRect fieldRect = borderRect.rect();
    FloatSize corner(5, 5);
    Path path;
    path.addRoundedRect(fieldRect, corner);
    fieldRect.inflate(-buttonBorderSize);
    corner.expand(-buttonBorderSize, -buttonBorderSize);
    path.addRoundedRect(fieldRect, corner);
    graphicsContext.setFillRule(WindRule::EvenOdd);
    graphicsContext.setFillColor(buttonBorderColor);
    graphicsContext.fillPath(path);
    path.clear();

    path.addRoundedRect(fieldRect, corner);
    graphicsContext.setFillRule(WindRule::NonZero);
    if (style.states.contains(ControlStyle::State::Pressed))
        graphicsContext.setFillColor(buttonBackgroundPressedColor);
    else if (style.states.contains(ControlStyle::State::Enabled)
        && style.states.contains(ControlStyle::State::Hovered))
        graphicsContext.setFillColor(buttonBackgroundHoveredColor);
    else
        graphicsContext.setFillColor(buttonBackgroundColor);
    graphicsContext.fillPath(path);

    if (style.states.contains(ControlStyle::State::Focused))
        Adwaita::paintFocus(graphicsContext, borderRect.rect(), buttonFocusOffset, systemFocusRingColor());

    if (!style.states.contains(ControlStyle::State::Enabled))
        graphicsContext.endTransparencyLayer();
}

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
