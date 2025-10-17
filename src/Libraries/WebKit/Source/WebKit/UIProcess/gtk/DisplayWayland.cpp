/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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

#if PLATFORM(WAYLAND)

#include <WebCore/GLContext.h>
#include <WebCore/GLDisplay.h>
#include <gtk/gtk.h>

// These includes need to be in this order because wayland-egl.h defines WL_EGL_PLATFORM
// and egl.h checks that to decide whether it's Wayland platform.
#include <wayland-egl.h>
#include <epoxy/egl.h>

#if USE(GTK4)
#include <gdk/wayland/gdkwayland.h>
#else
#include <gdk/gdkwayland.h>
#endif

namespace WebKit {
using namespace WebCore;

bool Display::isWayland() const
{
    return GDK_IS_WAYLAND_DISPLAY(m_gdkDisplay.get());
}

bool Display::initializeGLDisplayWayland() const
{
    if (!isWayland())
        return false;

#if USE(GTK4)
    m_glDisplay = GLDisplay::create(gdk_wayland_display_get_egl_display(m_gdkDisplay.get()));
#else
    auto* window = gtk_window_new(GTK_WINDOW_POPUP);
    gtk_widget_realize(window);
    if (auto context = adoptGRef(gdk_window_create_gl_context(gtk_widget_get_window(window), nullptr))) {
        gdk_gl_context_make_current(context.get());
        m_glDisplay = GLDisplay::create(eglGetCurrentDisplay());
    }
    gtk_widget_destroy(window);
#endif
    if (m_glDisplay)
        return true;

    auto* wlDisplay = gdk_wayland_display_get_wl_display(m_gdkDisplay.get());
    const char* extensions = eglQueryString(nullptr, EGL_EXTENSIONS);
    if (GLContext::isExtensionSupported(extensions, "EGL_KHR_platform_base"))
        m_glDisplay = GLDisplay::create(eglGetPlatformDisplay(EGL_PLATFORM_WAYLAND_KHR, wlDisplay, nullptr));
    if (!m_glDisplay && GLContext::isExtensionSupported(extensions, "EGL_EXT_platform_base"))
        m_glDisplay = GLDisplay::create(eglGetPlatformDisplayEXT(EGL_PLATFORM_WAYLAND_EXT, wlDisplay, nullptr));
    if (!m_glDisplay)
        m_glDisplay = GLDisplay::create(eglGetDisplay(wlDisplay));

    if (m_glDisplay) {
        m_glDisplayOwned = true;
        return true;
    }

    return false;
}

} // namespace WebKit

#endif // PLATFORM(WAYLAND)
