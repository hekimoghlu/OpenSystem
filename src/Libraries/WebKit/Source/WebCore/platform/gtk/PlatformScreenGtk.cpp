/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
#include "PlatformScreen.h"

#include "DestinationColorSpace.h"
#include "FloatRect.h"
#include "HostWindow.h"
#include "LocalFrameView.h"
#include "ScreenProperties.h"
#include "SystemSettings.h"
#include "Widget.h"
#include <gtk/gtk.h>

namespace WebCore {

static PlatformDisplayID widgetDisplayID(Widget* widget)
{
    if (!widget)
        return 0;

    auto* view = widget->root();
    if (!view)
        return 0;

    auto* hostWindow = view->hostWindow();
    if (!hostWindow)
        return 0;

    return hostWindow->displayID();
}

int screenDepth(Widget* widget)
{
    auto* data = screenData(widgetDisplayID(widget));
    return data ? data->screenDepth : 24;
}

int screenDepthPerComponent(Widget* widget)
{
    auto* data = screenData(widgetDisplayID(widget));
    return data ? data->screenDepthPerComponent : 8;
}

bool screenIsMonochrome(Widget* widget)
{
    return screenDepth(widget) < 2;
}

DestinationColorSpace screenColorSpace(Widget*)
{
    return DestinationColorSpace::SRGB();
}

bool screenHasInvertedColors()
{
    return false;
}

double fontDPI()
{
#if !USE(GTK4)
    // The code in this conditionally-compiled block is needed in order to
    // respect the GDK_DPI_SCALE setting that was present in GTK3 as an
    // additional font scaling factor.
    if (auto* display = gdk_display_get_default()) {
        if (auto* screen = gdk_display_get_default_screen(display))
            return gdk_screen_get_resolution(screen);
    }
#endif

    auto xftDPI = SystemSettings::singleton().xftDPI();
    if (xftDPI)
        return xftDPI.value() / 1024.0;

    auto* data = screenData(primaryScreenDisplayID());
    return data ? data->dpi : 96.;
}

double screenDPI(PlatformDisplayID screendisplayID)
{
    auto* data = screenData(screendisplayID);
    return data ? data->dpi : 96.;
}


FloatRect screenRect(Widget* widget)
{
    if (auto* data = screenData(widgetDisplayID(widget)))
        return data->screenRect;
    return { };
}

FloatRect screenAvailableRect(Widget* widget)
{
    if (auto* data = screenData(widgetDisplayID(widget)))
        return data->screenAvailableRect;
    return { };
}

bool screenSupportsExtendedColor(Widget*)
{
    return false;
}

#if ENABLE(TOUCH_EVENTS)
bool screenHasTouchDevice()
{
    auto* display = gdk_display_get_default();
    if (!display)
        return true;

    auto* seat = gdk_display_get_default_seat(display);
    return seat ? gdk_seat_get_capabilities(seat) & GDK_SEAT_CAPABILITY_TOUCH : true;
}

bool screenIsTouchPrimaryInputDevice()
{
    auto* display = gdk_display_get_default();
    if (!display)
        return true;

    auto* seat = gdk_display_get_default_seat(display);
    if (!seat)
        return true;

    auto* device = gdk_seat_get_pointer(seat);
    return device ? gdk_device_get_source(device) == GDK_SOURCE_TOUCHSCREEN : true;
}
#endif // ENABLE(TOUCH_EVENTS)

} // namespace WebCore
