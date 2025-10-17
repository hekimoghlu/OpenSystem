/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
#include "PointerLockManagerX11.h"

#if PLATFORM(X11)

#include "WebPageProxy.h"
#include <X11/Xlib.h>
#include <gtk/gtk.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/glib/GRefPtr.h>

#if USE(GTK4)
#include <gdk/x11/gdkx.h>
#else
#include <gdk/gdkx.h>
#endif

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PointerLockManagerX11);

PointerLockManagerX11::PointerLockManagerX11(WebPageProxy& webPage, const FloatPoint& position, const FloatPoint& globalPosition, WebMouseEventButton button, unsigned short buttons, OptionSet<WebEventModifier> modifiers)
    : PointerLockManager(webPage, position, globalPosition, button, buttons, modifiers)
{
}

bool PointerLockManagerX11::lock()
{
    if (!PointerLockManager::lock())
        return false;

    auto* viewWidget = m_webPage.viewWidget();
    auto* display = gtk_widget_get_display(viewWidget);
    auto* xDisplay = GDK_DISPLAY_XDISPLAY(gtk_widget_get_display(viewWidget));
#if USE(GTK4)
    GRefPtr<GdkCursor> cursor = adoptGRef(gdk_cursor_new_from_name("none", nullptr));
    auto window = GDK_SURFACE_XID(gtk_native_get_surface(gtk_widget_get_native(viewWidget)));
    auto xCursor = gdk_x11_display_get_xcursor(display, cursor.get());
#else
    GRefPtr<GdkCursor> cursor = adoptGRef(gdk_cursor_new_from_name(display, "none"));
    auto window = GDK_WINDOW_XID(gtk_widget_get_window(viewWidget));
    auto xCursor = gdk_x11_cursor_get_xcursor(cursor.get());
#endif
    int eventMask = PointerMotionMask | ButtonReleaseMask | ButtonPressMask | EnterWindowMask | LeaveWindowMask;
    XUngrabPointer(xDisplay, 0);
    return XGrabPointer(xDisplay, window, true, eventMask, GrabModeAsync, GrabModeAsync, window, xCursor, 0) == GrabSuccess;
}

bool PointerLockManagerX11::unlock()
{
    if (m_device)
        XUngrabPointer(GDK_DISPLAY_XDISPLAY(gtk_widget_get_display(m_webPage.viewWidget())), 0);

    return PointerLockManager::unlock();
}

void PointerLockManagerX11::didReceiveMotionEvent(const FloatPoint& point)
{
    auto delta = IntSize(point - m_initialPoint);
    if (delta.isZero())
        return;

    handleMotion(delta);
    auto* display = GDK_DISPLAY_XDISPLAY(gtk_widget_get_display(m_webPage.viewWidget()));
    float scaleFactor = m_webPage.deviceScaleFactor();
    IntSize warp = delta;
    warp.scale(-scaleFactor);
    XWarpPointer(display, None, None, 0, 0, 0, 0, warp.width(), warp.height());
}

} // namespace WebKit

#endif // PLATFORM(WAYLAND)
