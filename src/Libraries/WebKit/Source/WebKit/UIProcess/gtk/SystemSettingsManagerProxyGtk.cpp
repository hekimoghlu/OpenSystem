/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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
#include "SystemSettingsManagerProxy.h"

#include <gtk/gtk.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebKit {

String SystemSettingsManagerProxy::themeName() const
{
    if (auto* themeNameEnv = g_getenv("GTK_THEME")) {
        String name = String::fromUTF8(themeNameEnv);
        if (name.endsWithIgnoringASCIICase("-dark"_s) || name.endsWith(":dark"_s))
            return name.left(name.length() - 5);
        return name;
    }

    GUniqueOutPtr<char> themeNameSetting;
    g_object_get(m_settings, "gtk-theme-name", &themeNameSetting.outPtr(), nullptr);
    String name = String::fromUTF8(themeNameSetting.get());
    if (name.endsWithIgnoringASCIICase("-dark"_s))
        return name.left(name.length() - 5);
    return name;
}

bool SystemSettingsManagerProxy::darkMode() const
{
    gboolean preferDarkTheme;
    g_object_get(m_settings, "gtk-application-prefer-dark-theme", &preferDarkTheme, nullptr);
    if (preferDarkTheme)
        return true;

    // FIXME: These are just heuristics, we should get the dark mode from libhandy/libadwaita, falling back to the settings portal.
    // Or maybe just use the settings portal, because we don't want to depend on libhandy, and maybe don't want to depend on libadwaita?
    if (const char* themeNameEnv = g_getenv("GTK_THEME")) {
        String themeNameEnvString = String::fromUTF8(themeNameEnv);
        return themeNameEnvString.endsWith("-dark"_s) || themeNameEnvString.endsWith("-Dark"_s) || themeNameEnvString.endsWith(":dark"_s);
    }

    GUniqueOutPtr<char> themeName;
    g_object_get(m_settings, "gtk-theme-name", &themeName.outPtr(), nullptr);
    if (themeName) {
        String themeNameString = String::fromUTF8(themeName.get());
        if (themeNameString.endsWith("-dark"_s) || themeNameString.endsWith("-Dark"_s))
            return true;
    }

    return false;
}

String SystemSettingsManagerProxy::fontName() const
{
    GUniqueOutPtr<char> fontNameSetting;
    g_object_get(m_settings, "gtk-font-name", &fontNameSetting.outPtr(), nullptr);
    return String::fromUTF8(fontNameSetting.get());
}

int SystemSettingsManagerProxy::xftAntialias() const
{
    int antialiasSetting;
    g_object_get(m_settings, "gtk-xft-antialias", &antialiasSetting, nullptr);
    return antialiasSetting;
}

int SystemSettingsManagerProxy::xftHinting() const
{
    int hintingSetting;
    g_object_get(m_settings, "gtk-xft-hinting", &hintingSetting, nullptr);
    return hintingSetting;
}

String SystemSettingsManagerProxy::xftHintStyle() const
{
    GUniqueOutPtr<char> hintStyleSetting;
    g_object_get(m_settings, "gtk-xft-hintstyle", &hintStyleSetting.outPtr(), nullptr);
    return String::fromUTF8(hintStyleSetting.get());
}

String SystemSettingsManagerProxy::xftRGBA() const
{
    GUniqueOutPtr<char> rgbaSetting;
    g_object_get(m_settings, "gtk-xft-rgba", &rgbaSetting.outPtr(), nullptr);
    return String::fromUTF8(rgbaSetting.get());
}

int SystemSettingsManagerProxy::xftDPI() const
{
    int dpiSetting;
    g_object_get(m_settings, "gtk-xft-dpi", &dpiSetting, nullptr);
    return dpiSetting;
}

bool SystemSettingsManagerProxy::followFontSystemSettings() const
{
#if USE(GTK4)
#if GTK_CHECK_VERSION(4, 16, 0)
    GtkFontRendering fontRendering;
    g_object_get(m_settings, "gtk-font-rendering", &fontRendering, nullptr);
    return fontRendering == GTK_FONT_RENDERING_MANUAL;
#else
    return false;
#endif
#endif

    return true;
}

bool SystemSettingsManagerProxy::cursorBlink() const
{
    gboolean cursorBlinkSetting;
    g_object_get(m_settings, "gtk-cursor-blink", &cursorBlinkSetting, nullptr);
    return cursorBlinkSetting ? true : false;
}

int SystemSettingsManagerProxy::cursorBlinkTime() const
{
    int cursorBlinkTimeSetting;
    g_object_get(m_settings, "gtk-cursor-blink-time", &cursorBlinkTimeSetting, nullptr);
    return cursorBlinkTimeSetting;
}

bool SystemSettingsManagerProxy::primaryButtonWarpsSlider() const
{
    gboolean buttonSetting;
    g_object_get(m_settings, "gtk-primary-button-warps-slider", &buttonSetting, nullptr);
    return buttonSetting ? true : false;
}

bool SystemSettingsManagerProxy::overlayScrolling() const
{
    gboolean overlayScrollingSetting;
    g_object_get(m_settings, "gtk-overlay-scrolling", &overlayScrollingSetting, nullptr);
    return overlayScrollingSetting ? true : false;
}

bool SystemSettingsManagerProxy::enableAnimations() const
{
    gboolean enableAnimationsSetting;
    g_object_get(m_settings, "gtk-enable-animations", &enableAnimationsSetting, nullptr);
    return enableAnimationsSetting ? true : false;
}

SystemSettingsManagerProxy::SystemSettingsManagerProxy()
    : m_settings(gtk_settings_get_default())
{
    auto settingsChangedCallback = +[](SystemSettingsManagerProxy* settingsManager) {
        settingsManager->settingsDidChange();
    };

    g_signal_connect_swapped(m_settings, "notify::gtk-theme-name", G_CALLBACK(settingsChangedCallback), this);
    g_signal_connect_swapped(m_settings, "notify::gtk-font-name", G_CALLBACK(settingsChangedCallback), this);
    g_signal_connect_swapped(m_settings, "notify::gtk-xft-antialias", G_CALLBACK(settingsChangedCallback), this);
    g_signal_connect_swapped(m_settings, "notify::gtk-xft-dpi", G_CALLBACK(settingsChangedCallback), this);
    g_signal_connect_swapped(m_settings, "notify::gtk-xft-hinting", G_CALLBACK(settingsChangedCallback), this);
    g_signal_connect_swapped(m_settings, "notify::gtk-xft-hintstyle", G_CALLBACK(settingsChangedCallback), this);
    g_signal_connect_swapped(m_settings, "notify::gtk-xft-rgba", G_CALLBACK(settingsChangedCallback), this);
#if GTK_CHECK_VERSION(4, 16, 0)
    g_signal_connect_swapped(m_settings, "notify::gtk-font-rendering", G_CALLBACK(settingsChangedCallback), this);
#endif
    g_signal_connect_swapped(m_settings, "notify::gtk-cursor-blink", G_CALLBACK(settingsChangedCallback), this);
    g_signal_connect_swapped(m_settings, "notify::gtk-cursor-blink-time", G_CALLBACK(settingsChangedCallback), this);
    g_signal_connect_swapped(m_settings, "notify::gtk-primary-button-warps-slider", G_CALLBACK(settingsChangedCallback), this);
    g_signal_connect_swapped(m_settings, "notify::gtk-overlay-scrolling", G_CALLBACK(settingsChangedCallback), this);
    g_signal_connect_swapped(m_settings, "notify::gtk-enable-animations", G_CALLBACK(settingsChangedCallback), this);
    g_signal_connect_swapped(m_settings, "notify::gtk-application-prefer-dark-theme", G_CALLBACK(settingsChangedCallback), this);

    settingsDidChange();
}

} // namespace WebKit

