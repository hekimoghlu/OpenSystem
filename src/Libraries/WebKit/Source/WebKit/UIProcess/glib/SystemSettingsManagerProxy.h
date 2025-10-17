/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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

#include <wtf/NeverDestroyed.h>
#include <wtf/Noncopyable.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(GTK)
typedef struct _GtkSettings GtkSettings;
using PlatformSettings = GtkSettings;
#elif PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
typedef struct _WPESettings WPESettings;
using PlatformSettings = WPESettings;
#else
using PlatformSettings = void;
#endif

namespace WebKit {

class SystemSettingsManagerProxy {
    WTF_MAKE_NONCOPYABLE(SystemSettingsManagerProxy);
    friend NeverDestroyed<SystemSettingsManagerProxy>;

public:
    static void initialize();

private:
    SystemSettingsManagerProxy();

    void settingsDidChange();

    String themeName() const;
    bool darkMode() const;
    String fontName() const;
    int xftAntialias() const;
    int xftHinting() const;
    String xftHintStyle() const;
    String xftRGBA() const;
    int xftDPI() const;
    bool followFontSystemSettings() const;
    bool cursorBlink() const;
    int cursorBlinkTime() const;
    bool primaryButtonWarpsSlider() const;
    bool overlayScrolling() const;
    bool enableAnimations() const;

    PlatformSettings* m_settings { nullptr };
};

} // namespace WebKit
