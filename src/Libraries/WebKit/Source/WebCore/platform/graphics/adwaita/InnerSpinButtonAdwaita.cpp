/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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
#include "InnerSpinButtonAdwaita.h"

#include "GraphicsContextStateSaver.h"

#if USE(THEME_ADWAITA)

namespace WebCore {
using namespace WebCore::Adwaita;

InnerSpinButtonAdwaita::InnerSpinButtonAdwaita(ControlPart& part, ControlFactoryAdwaita& controlFactory)
    : ControlAdwaita(part, controlFactory)
{
}

void InnerSpinButtonAdwaita::draw(GraphicsContext& graphicsContext, const FloatRoundedRect& borderRect, float /*deviceScaleFactor*/, const ControlStyle& style)
{
    GraphicsContextStateSaver stateSaver(graphicsContext);

    SRGBA<uint8_t> spinButtonBorderColor;
    SRGBA<uint8_t> spinButtonBackgroundColor;
    SRGBA<uint8_t> spinButtonBackgroundHoveredColor;
    SRGBA<uint8_t> spinButtonBackgroundPressedColor;

    bool useDarkAppearance = style.states.contains(ControlStyle::State::DarkAppearance);
    if (useDarkAppearance) {
        spinButtonBorderColor = spinButtonBorderColorDark;
        spinButtonBackgroundColor = spinButtonBackgroundColorDark;
        spinButtonBackgroundHoveredColor = spinButtonBackgroundHoveredColorDark;
        spinButtonBackgroundPressedColor = spinButtonBackgroundPressedColorDark;
    } else {
        spinButtonBorderColor = spinButtonBorderColorLight;
        spinButtonBackgroundColor = spinButtonBackgroundColorLight;
        spinButtonBackgroundHoveredColor = spinButtonBackgroundHoveredColorLight;
        spinButtonBackgroundPressedColor = spinButtonBackgroundPressedColorLight;
    }

    FloatRect fieldRect = borderRect.rect();
    FloatSize corner(2, 2);
    Path path;
    path.addRoundedRect(fieldRect, corner);
    fieldRect.inflate(-buttonBorderSize);
    corner.expand(-buttonBorderSize, -buttonBorderSize);
    path.addRoundedRect(fieldRect, corner);
    graphicsContext.setFillRule(WindRule::EvenOdd);
    graphicsContext.setFillColor(spinButtonBorderColor);
    graphicsContext.fillPath(path);
    path.clear();

    path.addRoundedRect(fieldRect, corner);
    graphicsContext.setFillRule(WindRule::NonZero);
    graphicsContext.setFillColor(spinButtonBackgroundColor);
    graphicsContext.fillPath(path);
    path.clear();

    FloatRect buttonRect = fieldRect;
    buttonRect.setHeight(fieldRect.height() / 2.0);
    {
        if (style.states.contains(ControlStyle::State::SpinUp)) {
            path.addRoundedRect(FloatRoundedRect(buttonRect, corner, corner, { }, { }));
            if (style.states.contains(ControlStyle::State::Pressed))
                graphicsContext.setFillColor(spinButtonBackgroundPressedColor);
            else if (style.states.contains(ControlStyle::State::Hovered))
                graphicsContext.setFillColor(spinButtonBackgroundHoveredColor);
            graphicsContext.fillPath(path);
            path.clear();
        }

        Adwaita::paintArrow(graphicsContext, buttonRect, Adwaita::ArrowDirection::Up, useDarkAppearance);
    }

    buttonRect.move(0, buttonRect.height());
    {
        if (!style.states.contains(ControlStyle::State::SpinUp)) {
            path.addRoundedRect(FloatRoundedRect(buttonRect, { }, { }, corner, corner));
            if (style.states.contains(ControlStyle::State::Pressed))
                graphicsContext.setFillColor(spinButtonBackgroundPressedColor);
            else if (style.states.contains(ControlStyle::State::Hovered))
                graphicsContext.setFillColor(spinButtonBackgroundHoveredColor);
            else
                graphicsContext.setFillColor(spinButtonBackgroundColor);
            graphicsContext.fillPath(path);
            path.clear();
        }

        Adwaita::paintArrow(graphicsContext, buttonRect, Adwaita::ArrowDirection::Down, useDarkAppearance);
    }
}

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
