/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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

#if PLATFORM(X11)

#include <WebCore/GLContext.h>
#include <WebCore/GLDisplay.h>
#include <WebCore/XErrorTrapper.h>
#include <X11/Xatom.h>
#include <epoxy/egl.h>
#include <gtk/gtk.h>

#if USE(GTK4)
#include <gdk/x11/gdkx.h>
#else
#include <gdk/gdkx.h>
#endif

namespace WebKit {
using namespace WebCore;

bool Display::isX11() const
{
    return GDK_IS_X11_DISPLAY(m_gdkDisplay.get());
}

bool Display::initializeGLDisplayX11() const
{
    if (!isX11())
        return false;

#if USE(GTK4)
    m_glDisplay = GLDisplay::create(gdk_x11_display_get_egl_display(m_gdkDisplay.get()));
    if (m_glDisplay)
        return true;
#endif

    auto* xDisplay = GDK_DISPLAY_XDISPLAY(m_gdkDisplay.get());
    const char* extensions = eglQueryString(nullptr, EGL_EXTENSIONS);
    if (GLContext::isExtensionSupported(extensions, "EGL_KHR_platform_base"))
        m_glDisplay = GLDisplay::create(eglGetPlatformDisplay(EGL_PLATFORM_X11_KHR, xDisplay, nullptr));
    if (!m_glDisplay && GLContext::isExtensionSupported(extensions, "EGL_EXT_platform_base"))
        m_glDisplay = GLDisplay::create(eglGetPlatformDisplayEXT(EGL_PLATFORM_X11_KHR, xDisplay, nullptr));
    if (!m_glDisplay)
        m_glDisplay = GLDisplay::create(eglGetDisplay(xDisplay));

    if (m_glDisplay) {
        m_glDisplayOwned = true;
        return true;
    }

    return false;
}

String Display::accessibilityBusAddressX11() const
{
    auto* xDisplay = GDK_DISPLAY_XDISPLAY(m_gdkDisplay.get());
    Atom atspiBusAtom = XInternAtom(xDisplay, "AT_SPI_BUS", False);
    Atom type;
    int format;
    unsigned long itemCount, bytesAfter;
    unsigned char* data = nullptr;
    XErrorTrapper trapper(xDisplay, XErrorTrapper::Policy::Ignore);
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GTK port.
    XGetWindowProperty(xDisplay, RootWindowOfScreen(DefaultScreenOfDisplay(xDisplay)), atspiBusAtom, 0L, 8192, False, XA_STRING, &type, &format, &itemCount, &bytesAfter, &data);
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    auto atspiBusAddress = String::fromUTF8(reinterpret_cast<char*>(data));
    if (data)
        XFree(data);

    return atspiBusAddress;
}

} // namespace WebKit

#endif // PLATFORM(X11)
