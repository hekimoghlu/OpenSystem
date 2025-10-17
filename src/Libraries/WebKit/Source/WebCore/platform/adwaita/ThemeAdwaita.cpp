/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
#include "ThemeAdwaita.h"

#if USE(THEME_ADWAITA)

#include "Adwaita.h"
#include "Color.h"
#include "FontCascade.h"
#include "LengthSize.h"
#include <wtf/NeverDestroyed.h>

#if PLATFORM(GTK) || PLATFORM(WPE)
#include "SystemSettings.h"
#endif

namespace WebCore {
using namespace WebCore::Adwaita;

Theme& Theme::singleton()
{
    static NeverDestroyed<ThemeAdwaita> theme;
    return theme;
}

ThemeAdwaita::ThemeAdwaita()
{
#if PLATFORM(GTK) || PLATFORM(WPE)
    refreshSettings();

    // Note that Theme is NeverDestroy'd so the destructor will never be called to disconnect this.
    SystemSettings::singleton().addObserver([this](const SystemSettings::State& state) mutable {
        if (state.enableAnimations || state.themeName)
            this->refreshSettings();
    }, this);
#endif // PLATFORM(GTK) || PLATFORM(WPE)
}

#if PLATFORM(GTK) || PLATFORM(WPE)

void ThemeAdwaita::refreshSettings()
{
    if (auto enableAnimations = SystemSettings::singleton().enableAnimations())
        m_prefersReducedMotion = !enableAnimations.value();

    // For high contrast in GTK3 we can rely on the theme name and be accurate most of the time.
    // However whether or not high-contrast is enabled is also stored in GSettings/xdg-desktop-portal.
    // We could rely on libadwaita, dynamically, to re-use its logic.
#if PLATFORM(GTK) && !USE(GTK4)
    if (auto themeName = SystemSettings::singleton().themeName())
        m_prefersContrast = themeName == "HighContrast"_s || themeName == "HighContrastInverse"_s;
#endif // PLATFORM(GTK) && !USE(GTK4)
}

#endif // PLATFORM(GTK) || PLATFORM(WPE)

LengthSize ThemeAdwaita::controlSize(StyleAppearance appearance, const FontCascade& fontCascade, const LengthSize& zoomedSize, float zoomFactor) const
{
    if (!zoomedSize.width.isIntrinsicOrAuto() && !zoomedSize.height.isIntrinsicOrAuto())
        return Theme::controlSize(appearance, fontCascade, zoomedSize, zoomFactor);

    switch (appearance) {
    case StyleAppearance::Checkbox:
    case StyleAppearance::Radio: {
        LengthSize buttonSize = zoomedSize;
        if (buttonSize.width.isIntrinsicOrAuto())
            buttonSize.width = Length(12 * zoomFactor, LengthType::Fixed);
        if (buttonSize.height.isIntrinsicOrAuto())
            buttonSize.height = Length(12 * zoomFactor, LengthType::Fixed);
        return buttonSize;
    }
    case StyleAppearance::InnerSpinButton: {
        LengthSize spinButtonSize = zoomedSize;
        if (spinButtonSize.width.isIntrinsicOrAuto())
            spinButtonSize.width = Length(static_cast<int>(arrowSize * zoomFactor), LengthType::Fixed);
        if (spinButtonSize.height.isIntrinsicOrAuto() || fontCascade.size() > arrowSize)
            spinButtonSize.height = Length(fontCascade.size(), LengthType::Fixed);
        return spinButtonSize;
    }
    default:
        break;
    }

    return Theme::controlSize(appearance, fontCascade, zoomedSize, zoomFactor);
}

LengthSize ThemeAdwaita::minimumControlSize(StyleAppearance, const FontCascade&, const LengthSize& zoomedSize, float) const
{
    if (!zoomedSize.width.isIntrinsicOrAuto() && !zoomedSize.height.isIntrinsicOrAuto())
        return zoomedSize;

    LengthSize minSize = zoomedSize;
    if (minSize.width.isIntrinsicOrAuto())
        minSize.width = Length(0, LengthType::Fixed);
    if (minSize.height.isIntrinsicOrAuto())
        minSize.height = Length(0, LengthType::Fixed);
    return minSize;
}

LengthBox ThemeAdwaita::controlBorder(StyleAppearance appearance, const FontCascade& font, const LengthBox& zoomedBox, float zoomFactor) const
{
    switch (appearance) {
    case StyleAppearance::PushButton:
    case StyleAppearance::DefaultButton:
    case StyleAppearance::Button:
    case StyleAppearance::SquareButton:
        return zoomedBox;
    default:
        break;
    }

    return Theme::controlBorder(appearance, font, zoomedBox, zoomFactor);
}

void ThemeAdwaita::setAccentColor(const Color& color)
{
    if (m_accentColor == color)
        return;

    m_accentColor = color;

    platformColorsDidChange();
}

Color ThemeAdwaita::accentColor()
{
    return m_accentColor;
}

bool ThemeAdwaita::userPrefersContrast() const
{
#if !USE(GTK4)
    return m_prefersContrast;
#else
    return false;
#endif
}

bool ThemeAdwaita::userPrefersReducedMotion() const
{
    return m_prefersReducedMotion;
}

} // namespace WebCore

#endif // USE(THEME_ADWAITA)
