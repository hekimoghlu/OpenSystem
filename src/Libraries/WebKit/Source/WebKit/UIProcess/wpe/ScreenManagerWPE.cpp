/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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
#include "ScreenManager.h"

#if PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
#include <WebCore/PlatformScreen.h>
#include <cmath>
#include <wpe/wpe-platform.h>

namespace WebKit {
using namespace WebCore;

PlatformDisplayID ScreenManager::generatePlatformDisplayID(WPEScreen* screen)
{
    return wpe_screen_get_id(screen);
}

ScreenManager::ScreenManager()
{
    auto* display = wpe_display_get_primary();
    auto screensCount = wpe_display_get_n_screens(display);
    for (unsigned i = 0; i < screensCount; ++i) {
        if (auto* screen = wpe_display_get_screen(display, i))
            addScreen(screen);
    }

    g_signal_connect(display, "screen-added", G_CALLBACK(+[](WPEDisplay*, WPEScreen* screen, ScreenManager* manager) {
        manager->addScreen(screen);
        manager->updatePrimaryDisplayID();
        manager->propertiesDidChange();
    }), this);
    g_signal_connect(display, "screen-removed", G_CALLBACK(+[](WPEDisplay*, WPEScreen* screen, ScreenManager* manager) {
        manager->removeScreen(screen);
        manager->updatePrimaryDisplayID();
        manager->propertiesDidChange();
    }), this);
}

void ScreenManager::updatePrimaryDisplayID()
{
    // Assume the first screen is the primary one.
    auto* display = wpe_display_get_primary();
    auto screensCount = wpe_display_get_n_screens(display);
    auto* screen = screensCount ? wpe_display_get_screen(display, 0) : nullptr;
    m_primaryDisplayID = screen ? displayID(screen) : 0;
}

ScreenProperties ScreenManager::collectScreenProperties() const
{
    ScreenProperties properties;
    properties.primaryDisplayID = m_primaryDisplayID;

    for (const auto& iter : m_screenToDisplayIDMap) {
        WPEScreen* screen = iter.key;
        auto width = wpe_screen_get_width(screen);
        auto height = wpe_screen_get_height(screen);

        ScreenData data;
        data.screenRect = FloatRect(wpe_screen_get_x(screen), wpe_screen_get_y(screen), width, height);
        data.screenAvailableRect = data.screenRect;
        data.screenDepth = 24;
        data.screenDepthPerComponent = 8;
        data.screenSize = { wpe_screen_get_physical_width(screen), wpe_screen_get_physical_height(screen) };
        static constexpr double millimetresPerInch = 25.4;
        double diagonalInPixels = std::hypot(width, height);
        double diagonalInInches = std::hypot(data.screenSize.width(), data.screenSize.height()) / millimetresPerInch;
        data.dpi = diagonalInPixels / diagonalInInches;
        properties.screenDataMap.add(iter.value, WTFMove(data));
    }

    // FIXME: don't use PlatformScreen from the UI process, better use ScreenManager directly.
    WebCore::setScreenProperties(properties);

    return properties;
}

} // namespace WebKit

#endif // PLATFORM(WPE) && ENABLE(WPE_PLATFORM)
