/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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
#include "ToggleButtonAdwaita.h"

#if USE(THEME_ADWAITA)

#include "ColorBlending.h"
#include "GraphicsContextStateSaver.h"

namespace WebCore {
using namespace WebCore::Adwaita;

ToggleButtonAdwaita::ToggleButtonAdwaita(ControlPart& part, ControlFactoryAdwaita& controlFactory)
    : ControlAdwaita(part, controlFactory)
{
}

void ToggleButtonAdwaita::draw(GraphicsContext& graphicsContext, const FloatRoundedRect& borderRect, float deviceScaleFactor, const ControlStyle& style)
{
    if (m_owningPart.type() == StyleAppearance::Checkbox)
        drawCheckbox(graphicsContext, borderRect, deviceScaleFactor, style);
    else
        drawRadio(graphicsContext, borderRect, deviceScaleFactor, style);
}

void ToggleButtonAdwaita::drawCheckbox(GraphicsContext& graphicsContext, const FloatRoundedRect& borderRect, float /*deviceScaleFactor*/, const ControlStyle& style)
{
    GraphicsContextStateSaver stateSaver(graphicsContext);

    FloatRect fieldRect = borderRect.rect();
    if (fieldRect.width() != fieldRect.height()) {
        auto buttonSize = std::min(fieldRect.width(), fieldRect.height());
        fieldRect.setSize({ buttonSize, buttonSize });
        if (fieldRect.width() != borderRect.rect().width())
            fieldRect.move((borderRect.rect().width() - fieldRect.width()) / 2.0, 0);
        else
            fieldRect.move(0, (borderRect.rect().height() - fieldRect.height()) / 2.0);
    }

    SRGBA<uint8_t> toggleBorderColor;
    SRGBA<uint8_t> toggleBorderHoverColor;

    if (style.states.contains(ControlStyle::State::DarkAppearance)) {
        toggleBorderColor = toggleBorderColorDark;
        toggleBorderHoverColor = toggleBorderHoveredColorDark;
    } else {
        toggleBorderColor = toggleBorderColorLight;
        toggleBorderHoverColor = toggleBorderHoveredColorLight;
    }

    Color accentColor = this->accentColor(style);
    Color foregroundColor = accentColor.luminance() > 0.5 ? Color(SRGBA<uint8_t> { 0, 0, 0, 204 }) : Color::white;
    Color accentHoverColor = blendSourceOver(accentColor, foregroundColor.colorWithAlphaMultipliedBy(0.1));

    if (!style.states.contains(ControlStyle::State::Enabled))
        graphicsContext.beginTransparencyLayer(disabledOpacity);

    FloatSize corner(2, 2);
    Path path;

    if (style.states.containsAny({ ControlStyle::State::Checked, ControlStyle::State::Indeterminate })) {
        path.addRoundedRect(fieldRect, corner);
        graphicsContext.setFillRule(WindRule::NonZero);
        if (style.states.contains(ControlStyle::State::Hovered) && style.states.contains(ControlStyle::State::Enabled))
            graphicsContext.setFillColor(accentHoverColor);
        else
            graphicsContext.setFillColor(accentColor);
        graphicsContext.fillPath(path);
        path.clear();

        GraphicsContextStateSaver checkedStateSaver(graphicsContext);
        graphicsContext.translate(fieldRect.x(), fieldRect.y());
        graphicsContext.scale(FloatSize::narrowPrecision(fieldRect.width() / toggleSize, fieldRect.height() / toggleSize));
        if (style.states.contains(ControlStyle::State::Indeterminate))
            path.addRoundedRect(FloatRect(2, 5, 10, 4), corner);
        else {
            path.moveTo({ 2.43, 6.57 });
            path.addLineTo({ 7.5, 11.63 });
            path.addLineTo({ 14, 5 });
            path.addLineTo({ 14, 1 });
            path.addLineTo({ 7.5, 7.38 });
            path.addLineTo({ 4.56, 4.44 });
            path.closeSubpath();
        }

        graphicsContext.setFillColor(foregroundColor);
        graphicsContext.fillPath(path);
    } else {
        path.addRoundedRect(fieldRect, corner);
        if (style.states.contains(ControlStyle::State::Hovered) && style.states.contains(ControlStyle::State::Enabled))
            graphicsContext.setFillColor(toggleBorderHoverColor);
        else
            graphicsContext.setFillColor(toggleBorderColor);
        graphicsContext.fillPath(path);
        path.clear();

        fieldRect.inflate(-toggleBorderSize);
        corner.expand(-buttonBorderSize, -buttonBorderSize);
        path.addRoundedRect(fieldRect, corner);
        graphicsContext.setFillColor(foregroundColor);
        graphicsContext.fillPath(path);
    }

    if (style.states.contains(ControlStyle::State::Focused))
        Adwaita::paintFocus(graphicsContext, borderRect.rect(), toggleFocusOffset, focusColor(accentColor));

    if (!style.states.contains(ControlStyle::State::Enabled))
        graphicsContext.endTransparencyLayer();
}

void ToggleButtonAdwaita::drawRadio(GraphicsContext& graphicsContext, const FloatRoundedRect& borderRect, float /*deviceScaleFactor*/, const ControlStyle& style)
{
    GraphicsContextStateSaver stateSaver(graphicsContext);
    FloatRect fieldRect = borderRect.rect();
    if (fieldRect.width() != fieldRect.height()) {
        auto buttonSize = std::min(fieldRect.width(), fieldRect.height());
        fieldRect.setSize({ buttonSize, buttonSize });
        if (fieldRect.width() != borderRect.rect().width())
            fieldRect.move((borderRect.rect().width() - fieldRect.width()) / 2.0, 0);
        else
            fieldRect.move(0, (borderRect.rect().height() - fieldRect.height()) / 2.0);
    }

    SRGBA<uint8_t> toggleBorderColor;
    SRGBA<uint8_t> toggleBorderHoverColor;

    if (style.states.contains(ControlStyle::State::DarkAppearance)) {
        toggleBorderColor = toggleBorderColorDark;
        toggleBorderHoverColor = toggleBorderHoveredColorDark;
    } else {
        toggleBorderColor = toggleBorderColorLight;
        toggleBorderHoverColor = toggleBorderHoveredColorLight;
    }

    Color accentColor = this->accentColor(style);
    Color foregroundColor = accentColor.luminance() > 0.5 ? Color(SRGBA<uint8_t> { 0, 0, 0, 204 }) : Color::white;
    Color accentHoverColor = blendSourceOver(accentColor, foregroundColor.colorWithAlphaMultipliedBy(0.1));

    if (!style.states.contains(ControlStyle::State::Enabled))
        graphicsContext.beginTransparencyLayer(disabledOpacity);

    Path path;

    if (style.states.containsAny({ ControlStyle::State::Checked, ControlStyle::State::Indeterminate })) {
        path.addEllipseInRect(fieldRect);
        graphicsContext.setFillRule(WindRule::NonZero);
        if (style.states.contains(ControlStyle::State::Hovered) && style.states.contains(ControlStyle::State::Enabled))
            graphicsContext.setFillColor(accentHoverColor);
        else
            graphicsContext.setFillColor(accentColor);
        graphicsContext.fillPath(path);
        path.clear();

        fieldRect.inflate(-(fieldRect.width() - fieldRect.width() * 0.70));
        path.addEllipseInRect(fieldRect);
        graphicsContext.setFillColor(foregroundColor);
        graphicsContext.fillPath(path);
    } else {
        path.addEllipseInRect(fieldRect);
        if (style.states.contains(ControlStyle::State::Hovered) && style.states.contains(ControlStyle::State::Enabled))
            graphicsContext.setFillColor(toggleBorderHoverColor);
        else
            graphicsContext.setFillColor(toggleBorderColor);
        graphicsContext.fillPath(path);
        path.clear();

        fieldRect.inflate(-toggleBorderSize);
        path.addEllipseInRect(fieldRect);
        graphicsContext.setFillColor(foregroundColor);
        graphicsContext.fillPath(path);
    }

    if (style.states.contains(ControlStyle::State::Focused))
        Adwaita::paintFocus(graphicsContext, borderRect.rect(), toggleFocusOffset, focusColor(accentColor), Adwaita::PaintRounded::Yes);

    if (!style.states.contains(ControlStyle::State::Enabled))
        graphicsContext.endTransparencyLayer();
}

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
