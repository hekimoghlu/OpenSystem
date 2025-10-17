/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#include "SystemSettings.h"

#if PLATFORM(GTK) || PLATFORM(WPE)

namespace WebCore {

SystemSettings& SystemSettings::singleton()
{
    static NeverDestroyed<SystemSettings> settings;
    return settings;
}

SystemSettings::SystemSettings() = default;

void SystemSettings::updateSettings(const SystemSettings::State& state)
{
    if (state.themeName)
        m_state.themeName = state.themeName;

    if (state.darkMode)
        m_state.darkMode = state.darkMode;

    if (state.fontName)
        m_state.fontName = state.fontName;

    if (state.xftHinting)
        m_state.xftHinting = state.xftHinting;

    if (state.xftHintStyle)
        m_state.xftHintStyle = state.xftHintStyle;

    if (state.xftAntialias)
        m_state.xftAntialias = state.xftAntialias;

    if (state.xftRGBA)
        m_state.xftRGBA = state.xftRGBA;

    if (state.xftDPI)
        m_state.xftDPI = state.xftDPI;

    if (state.followFontSystemSettings)
        m_state.followFontSystemSettings = state.followFontSystemSettings;

    if (state.cursorBlink)
        m_state.cursorBlink = state.cursorBlink;

    if (state.cursorBlinkTime)
        m_state.cursorBlinkTime = state.cursorBlinkTime;

    if (state.primaryButtonWarpsSlider)
        m_state.primaryButtonWarpsSlider = state.primaryButtonWarpsSlider;

    if (state.overlayScrolling)
        m_state.overlayScrolling = state.overlayScrolling;

    if (state.enableAnimations)
        m_state.enableAnimations = state.enableAnimations;

    for (auto* context : copyToVector(m_observers.keys())) {
        const auto it = m_observers.find(context);
        if (it == m_observers.end())
            continue;
        it->value(state);
    }
}

std::optional<FontRenderOptions::Hinting> SystemSettings::hintStyle() const
{
    std::optional<FontRenderOptions::Hinting> hintStyle;
    if (m_state.xftHinting && !m_state.xftHinting.value())
        hintStyle = FontRenderOptions::Hinting::None;
    else if (m_state.xftHinting == 1) {
        if (m_state.xftHintStyle == "hintnone"_s)
            hintStyle = FontRenderOptions::Hinting::None;
        else if (m_state.xftHintStyle == "hintslight"_s)
            hintStyle = FontRenderOptions::Hinting::Slight;
        else if (m_state.xftHintStyle == "hintmedium"_s)
            hintStyle = FontRenderOptions::Hinting::Medium;
        else if (m_state.xftHintStyle == "hintfull"_s)
            hintStyle = FontRenderOptions::Hinting::Full;
    }

    return hintStyle;
}

std::optional<FontRenderOptions::SubpixelOrder> SystemSettings::subpixelOrder() const
{
    std::optional<FontRenderOptions::SubpixelOrder> subpixelOrder;
    if (m_state.xftRGBA) {
        if (m_state.xftRGBA == "rgb"_s)
            subpixelOrder = FontRenderOptions::SubpixelOrder::HorizontalRGB;
        else if (m_state.xftRGBA == "bgr"_s)
            subpixelOrder = FontRenderOptions::SubpixelOrder::HorizontalBGR;
        else if (m_state.xftRGBA == "vrgb"_s)
            subpixelOrder = FontRenderOptions::SubpixelOrder::VerticalRGB;
        else if (m_state.xftRGBA == "vbgr"_s)
            subpixelOrder = FontRenderOptions::SubpixelOrder::VerticalBGR;
    }
    return subpixelOrder;
}

std::optional<FontRenderOptions::Antialias> SystemSettings::antialiasMode() const
{
    std::optional<FontRenderOptions::Antialias> antialiasMode;
    if (m_state.xftAntialias && !m_state.xftAntialias.value())
        antialiasMode = FontRenderOptions::Antialias::None;
    else if (m_state.xftAntialias == 1)
        antialiasMode = subpixelOrder().has_value() ? FontRenderOptions::Antialias::Subpixel : FontRenderOptions::Antialias::Normal;

    return antialiasMode;
}

String SystemSettings::defaultSystemFont() const
{
    auto fontDescription = fontName();
    if (!fontDescription || fontDescription->isEmpty())
        return "Sans"_s;

    // We need to remove the size from the value of the property,
    // which is separated from the font family using a space.
    if (auto index = fontDescription->reverseFind(' '); index != notFound)
        fontDescription = fontDescription->left(index);
    return *fontDescription;
}

void SystemSettings::addObserver(Function<void(const SystemSettings::State&)>&& handler, void* context)
{
    m_observers.add(context, WTFMove(handler));
}

void SystemSettings::removeObserver(void* context)
{
    m_observers.remove(context);
}

} // namespace WebCore

#endif // PLATFORM(GTK) || PLATFORM(WPE)
