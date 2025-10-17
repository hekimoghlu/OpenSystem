/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#include "MenuListAdwaita.h"

#include "ButtonControlAdwaita.h"
#include "GraphicsContextStateSaver.h"

#if USE(THEME_ADWAITA)

namespace WebCore {
using namespace WebCore::Adwaita;

MenuListAdwaita::MenuListAdwaita(ControlPart& part, ControlFactoryAdwaita& controlFactory)
    : ControlAdwaita(part, controlFactory)
{
}

void MenuListAdwaita::draw(GraphicsContext& graphicsContext, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    GraphicsContextStateSaver stateSaver(graphicsContext);

    ButtonControlAdwaita::drawButton(graphicsContext, borderRect, deviceScaleFactor, style);

    auto zoomedArrowSize = menuListButtonArrowSize * style.zoomFactor;
    FloatRect fieldRect = borderRect.rect();
    fieldRect.inflate(menuListButtonBorderSize);
    if (style.states.contains(ControlStyle::State::InlineFlippedWritingMode))
        fieldRect.move(menuListButtonPadding, 0);
    else
        fieldRect.move(fieldRect.width() - (zoomedArrowSize + menuListButtonPadding), 0);
    fieldRect.setWidth(zoomedArrowSize);
    Adwaita::paintArrow(graphicsContext, fieldRect, Adwaita::ArrowDirection::Down, style.states.contains(ControlStyle::State::DarkAppearance));

    if (style.states.contains(ControlStyle::State::Focused))
        Adwaita::paintFocus(graphicsContext, borderRect.rect(), menuListButtonFocusOffset, systemFocusRingColor());
}

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
