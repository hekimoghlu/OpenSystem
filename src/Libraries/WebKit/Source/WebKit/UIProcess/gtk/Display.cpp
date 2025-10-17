/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
#include "Display.h"

#include "SystemSettingsManagerProxy.h"
#include <WebCore/GLContext.h>
#include <WebCore/GLDisplay.h>
#include <WebCore/GtkVersioning.h>
#include <epoxy/egl.h>
#include <gtk/gtk.h>
#include <mutex>
#include <wtf/NeverDestroyed.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebKit {
using namespace WebCore;

Display& Display::singleton()
{
    static LazyNeverDestroyed<Display> display;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        display.construct();
    });
    return display;
}

Display::Display()
{
    if (!gtk_init_check(nullptr, nullptr))
        return;

    // As soon as gtk is initialized we listen to GtkSettings.
    SystemSettingsManagerProxy::initialize();

    m_gdkDisplay = gdk_display_manager_get_default_display(gdk_display_manager_get());
    if (!m_gdkDisplay)
        return;

    g_signal_connect(m_gdkDisplay.get(), "closed", G_CALLBACK(+[](GdkDisplay*, gboolean, gpointer userData) {
        auto& display = *static_cast<Display*>(userData);
        if (display.m_glDisplay && display.m_glDisplayOwned)
            display.m_glDisplay->terminate();
        display.m_glDisplay = nullptr;
    }), this);
}

Display::~Display()
{
    if (m_gdkDisplay)
        g_signal_handlers_disconnect_by_data(m_gdkDisplay.get(), this);
}

GLDisplay* Display::glDisplay() const
{
    if (m_glInitialized)
        return m_glDisplay.get();

    m_glInitialized = true;
    if (!m_gdkDisplay)
        return nullptr;

#if PLATFORM(X11)
    if (initializeGLDisplayX11())
        return m_glDisplay.get();
#endif
#if PLATFORM(WAYLAND)
    if (initializeGLDisplayWayland())
        return m_glDisplay.get();
#endif

    return nullptr;
}

String Display::accessibilityBusAddress() const
{
    if (!m_gdkDisplay)
        return { };

#if USE(GTK4)
    if (const char* atspiBusAddress = static_cast<const char*>(g_object_get_data(G_OBJECT(m_gdkDisplay.get()), "-gtk-atspi-bus-address")))
        return String::fromUTF8(atspiBusAddress);
#endif

#if PLATFORM(X11)
    if (isX11())
        return accessibilityBusAddressX11();
#endif

    return { };
}

#if !PLATFORM(X11)
bool Display::isX11() const
{
    return false;
}
#endif

#if !PLATFORM(WAYLAND)
bool Display::isWayland() const
{
    return false;
}
#endif

} // namespace WebKit
