/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
#pragma once

#if PLATFORM(GTK) || PLATFORM(WPE)

#include "FontRenderOptions.h"
#include <optional>
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct SystemSettingsState {
    std::optional<String> themeName;
    std::optional<bool> darkMode;
    std::optional<String> fontName;
    std::optional<int> xftAntialias;
    std::optional<int> xftHinting;
    std::optional<String> xftHintStyle;
    std::optional<String> xftRGBA;
    std::optional<int> xftDPI;
    std::optional<bool> followFontSystemSettings;
    std::optional<bool> cursorBlink;
    std::optional<int> cursorBlinkTime;
    std::optional<bool> primaryButtonWarpsSlider;
    std::optional<bool> overlayScrolling;
    std::optional<bool> enableAnimations;
};

class SystemSettings {
    WTF_MAKE_NONCOPYABLE(SystemSettings);
    friend NeverDestroyed<SystemSettings>;
public:
    static SystemSettings& singleton();

    using State = SystemSettingsState;

    void updateSettings(const SystemSettings::State&);

    const State& settingsState() const { return m_state; }

    void addObserver(Function<void(const State&)>&&, void* context);
    void removeObserver(void* context);

    std::optional<String> themeName() const { return m_state.themeName; }
    std::optional<bool> darkMode() const { return m_state.darkMode; }
    std::optional<String> fontName() const { return m_state.fontName; }
    std::optional<bool> cursorBlink() const { return m_state.cursorBlink; }
    std::optional<int> cursorBlinkTime() const { return m_state.cursorBlinkTime; }
    std::optional<int> xftDPI() const { return m_state.xftDPI; }
    std::optional<bool> followFontSystemSettings() const { return m_state.followFontSystemSettings; }
    std::optional<bool> overlayScrolling() const { return m_state.overlayScrolling; }
    std::optional<bool> primaryButtonWarpsSlider() const { return m_state.primaryButtonWarpsSlider; }
    std::optional<bool> enableAnimations() const { return m_state.enableAnimations; }

    std::optional<FontRenderOptions::Hinting> hintStyle() const;
    std::optional<FontRenderOptions::SubpixelOrder> subpixelOrder() const;
    std::optional<FontRenderOptions::Antialias> antialiasMode() const;

    String defaultSystemFont() const;

private:
    SystemSettings();

    State m_state;
    UncheckedKeyHashMap<void*, Function<void(const State&)>> m_observers;
};

} // namespace WebCore

#endif // PLATFORM(GTK) || PLATFORM(WPE)

