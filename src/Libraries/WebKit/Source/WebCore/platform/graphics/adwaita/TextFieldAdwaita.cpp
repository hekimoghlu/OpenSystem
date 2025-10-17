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
#include "config.h"
#include "TextFieldAdwaita.h"

#include "GraphicsContextStateSaver.h"

#if USE(THEME_ADWAITA)

namespace WebCore {
using namespace WebCore::Adwaita;

TextFieldAdwaita::TextFieldAdwaita(ControlPart& part, ControlFactoryAdwaita& controlFactory)
    : ControlAdwaita(part, controlFactory)
{
}

void TextFieldAdwaita::draw(GraphicsContext& graphicsContext, const FloatRoundedRect& borderRect, float /*deviceScaleFactor*/, const ControlStyle& style)
{
    GraphicsContextStateSaver stateSaver(graphicsContext);

    SRGBA<uint8_t> textFieldBackgroundColor;
    SRGBA<uint8_t> textFieldBorderColor;

    if (style.states.contains(ControlStyle::State::DarkAppearance)) {
        textFieldBackgroundColor = textFieldBackgroundColorDark;
        textFieldBorderColor= textFieldBorderColorDark;
    } else {
        textFieldBackgroundColor = textFieldBackgroundColorLight;
        textFieldBorderColor = textFieldBorderColorLight;
    }

    bool enabled = style.states.contains(ControlStyle::State::Enabled) && !style.states.contains(ControlStyle::State::ReadOnly);
    int borderSize = textFieldBorderSize;
    if (style.states.contains(ControlStyle::State::Focused))
        borderSize *= 2;

    if (!enabled)
        graphicsContext.beginTransparencyLayer(disabledOpacity);

    FloatRect fieldRect = borderRect.rect();
    FloatSize corner(5, 5);
    Path path;
    path.addRoundedRect(fieldRect, corner);
    fieldRect.inflate(-borderSize);
    corner.expand(-borderSize, -borderSize);
    path.addRoundedRect(fieldRect, corner);
    graphicsContext.setFillRule(WindRule::EvenOdd);
    if (enabled && style.states.contains(ControlStyle::State::Focused))
        graphicsContext.setFillColor(systemFocusRingColor());
    else
        graphicsContext.setFillColor(textFieldBorderColor);
    graphicsContext.fillPath(path);
    path.clear();

    path.addRoundedRect(fieldRect, corner);
    graphicsContext.setFillRule(WindRule::NonZero);
    graphicsContext.setFillColor(textFieldBackgroundColor);
    graphicsContext.fillPath(path);

    if (style.states.contains(ControlStyle::State::ListButton)) {
        auto zoomedArrowSize = menuListButtonArrowSize * style.zoomFactor;
        FloatRect arrowRect = borderRect.rect();
        if (style.states.contains(ControlStyle::State::InlineFlippedWritingMode))
            arrowRect.move(textFieldBorderSize * 2, 0);
        else
            arrowRect.move(arrowRect.width() - (zoomedArrowSize + textFieldBorderSize * 2), 0);
        arrowRect.setWidth(zoomedArrowSize);
        bool useDarkAppearance = style.states.contains(ControlStyle::State::DarkAppearance);
        Adwaita::paintArrow(graphicsContext, arrowRect, Adwaita::ArrowDirection::Down, useDarkAppearance);
    }

    if (!enabled)
        graphicsContext.endTransparencyLayer();
}

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
