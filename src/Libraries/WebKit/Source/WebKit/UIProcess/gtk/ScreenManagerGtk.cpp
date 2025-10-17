/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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

#include <WebCore/GtkUtilities.h>
#include <WebCore/PlatformScreen.h>
#include <cmath>

namespace WebKit {
using namespace WebCore;

PlatformDisplayID ScreenManager::generatePlatformDisplayID(GdkMonitor*)
{
    static PlatformDisplayID id;
    return ++id;
}

ScreenManager::ScreenManager()
{
    auto* display = gdk_display_get_default();
#if USE(GTK4)
    auto* monitors = gdk_display_get_monitors(display);
    auto monitorsCount = g_list_model_get_n_items(monitors);
    for (unsigned i = 0; i < monitorsCount; ++i) {
        auto monitor = adoptGRef(GDK_MONITOR(g_list_model_get_item(monitors, i)));
        addScreen(monitor.get());
    }
    g_signal_connect(monitors, "items-changed", G_CALLBACK(+[](GListModel* monitors, guint index, guint removedCount, guint addedCount, ScreenManager* manager) {
        for (unsigned i = 0; i < removedCount; ++i)
            manager->removeScreen(manager->m_screens[index].get());
        for (unsigned i = 0; i < addedCount; ++i) {
            auto monitor = adoptGRef(GDK_MONITOR(g_list_model_get_item(monitors, index + i)));
            manager->addScreen(monitor.get());
        }
        manager->updatePrimaryDisplayID();
        manager->propertiesDidChange();
    }), this);
#else
    auto monitorsCount = gdk_display_get_n_monitors(display);
    for (int i = 0; i < monitorsCount; ++i) {
        if (auto* monitor = gdk_display_get_monitor(display, i))
            addScreen(monitor);
    }
    g_signal_connect(display, "monitor-added", G_CALLBACK(+[](GdkDisplay*, GdkMonitor* monitor, ScreenManager* manager) {
        manager->addScreen(monitor);
        manager->updatePrimaryDisplayID();
        manager->propertiesDidChange();
    }), this);
    g_signal_connect(display, "monitor-removed", G_CALLBACK(+[](GdkDisplay*, GdkMonitor* monitor, ScreenManager* manager) {
        manager->removeScreen(monitor);
        manager->updatePrimaryDisplayID();
        manager->propertiesDidChange();
    }), this);
#endif
    updatePrimaryDisplayID();
}

void ScreenManager::updatePrimaryDisplayID()
{
    auto* display = gdk_display_get_default();
#if USE(GTK4)
    // GTK4 doesn't have the concept of primary monitor, so we always use the first one.
    auto* monitors = gdk_display_get_monitors(display);
    if (!g_list_model_get_n_items(monitors)) {
        m_primaryDisplayID = 0;
        return;
    }

    auto monitor = adoptGRef(GDK_MONITOR(g_list_model_get_item(monitors, 0)));
    m_primaryDisplayID = displayID(monitor.get());
#else
    auto* primaryMonitor = gdk_display_get_primary_monitor(display);
    if (!primaryMonitor) {
        if (gdk_display_get_n_monitors(display))
            primaryMonitor = gdk_display_get_monitor(display, 0);
    }
    m_primaryDisplayID = primaryMonitor ? displayID(primaryMonitor) : 0;
#endif
}

ScreenProperties ScreenManager::collectScreenProperties() const
{
#if !USE(GTK4)
    auto systemVisual = [](GdkDisplay* display) -> GdkVisual* {
        if (auto* screen = gdk_display_get_default_screen(display))
            return gdk_screen_get_system_visual(screen);

        return nullptr;
    };

    auto* display = gdk_display_get_default();
#endif

    ScreenProperties properties;
    properties.primaryDisplayID = m_primaryDisplayID;

    for (const auto& iter : m_screenToDisplayIDMap) {
        GdkMonitor* monitor = iter.key;
        ScreenData data;
        GdkRectangle workArea;
        monitorWorkArea(monitor, &workArea);
        data.screenAvailableRect = FloatRect(workArea.x, workArea.y, workArea.width, workArea.height);
        GdkRectangle geometry;
        gdk_monitor_get_geometry(monitor, &geometry);
        data.screenRect = FloatRect(geometry.x, geometry.y, geometry.width, geometry.height);
#if USE(GTK4)
        data.screenDepth = 24;
        data.screenDepthPerComponent = 8;
#else
        auto* visual = systemVisual(display);
        data.screenDepth = visual ? gdk_visual_get_depth(visual) : 24;
        if (visual) {
            int redDepth;
            gdk_visual_get_red_pixel_details(visual, nullptr, nullptr, &redDepth);
            data.screenDepthPerComponent = redDepth;
        } else
            data.screenDepthPerComponent = 8;
#endif
        data.screenSize = { gdk_monitor_get_width_mm(monitor), gdk_monitor_get_height_mm(monitor) };
        static const double millimetresPerInch = 25.4;
        double diagonalInPixels = std::hypot(geometry.width, geometry.height);
        double diagonalInInches = std::hypot(data.screenSize.width(), data.screenSize.height()) / millimetresPerInch;
        data.dpi = diagonalInPixels / diagonalInInches;
        properties.screenDataMap.add(iter.value, WTFMove(data));
    }

    // FIXME: don't use PlatformScreen from the UI process, better use ScreenManager directly.
    WebCore::setScreenProperties(properties);

    return properties;
}

} // namespace WebKit
