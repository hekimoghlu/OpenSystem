/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
#include "NotImplemented.h"
#include "ScreenProperties.h"
#include "Widget.h"

namespace WebCore {

#if ENABLE(WPE_PLATFORM)
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
#endif

int screenDepth(Widget* widget)
{
#if ENABLE(WPE_PLATFORM)
    auto* data = screenData(widgetDisplayID(widget));
    return data ? data->screenDepth : 24;
#else
    UNUSED_PARAM(widget);
    notImplemented();
    return 24;
#endif
}

int screenDepthPerComponent(Widget* widget)
{
#if ENABLE(WPE_PLATFORM)
    auto* data = screenData(widgetDisplayID(widget));
    return data ? data->screenDepthPerComponent : 8;
#else
    UNUSED_PARAM(widget);
    notImplemented();
    return 8;
#endif
}

bool screenIsMonochrome(Widget* widget)
{
#if ENABLE(WPE_PLATFORM)
    return screenDepth(widget) < 2;
#else
    UNUSED_PARAM(widget);
    notImplemented();
    return false;
#endif
}

bool screenHasInvertedColors()
{
    return false;
}

double screenDPI(PlatformDisplayID screendisplayID)
{
#if ENABLE(WPE_PLATFORM)
    auto* data = screenData(screendisplayID);
    return data ? data->dpi : 96.;
#else
    UNUSED_PARAM(screendisplayID);
    notImplemented();
    return 96;
#endif
}

double fontDPI()
{
#if ENABLE(WPE_PLATFORM)
    // In WPE, there is no notion of font scaling separate from device DPI.
    return screenDPI(primaryScreenDisplayID());
#else
    notImplemented();
    return 96.;
#endif
}

FloatRect screenRect(Widget* widget)
{
#if ENABLE(WPE_PLATFORM)
    if (auto* data = screenData(widgetDisplayID(widget)))
        return data->screenRect;
#endif
    // WPE can't offer any more useful information about the screen size,
    // so we use the Widget's bounds rectangle (size of which equals the WPE view size).

    if (!widget)
        return { };
    return widget->boundsRect();
}

FloatRect screenAvailableRect(Widget* widget)
{
#if ENABLE(WPE_PLATFORM)
    if (auto* data = screenData(widgetDisplayID(widget)))
        return data->screenAvailableRect;
#endif
    return screenRect(widget);
}

DestinationColorSpace screenColorSpace(Widget*)
{
    return DestinationColorSpace::SRGB();
}

bool screenSupportsExtendedColor(Widget*)
{
    return false;
}

#if ENABLE(TOUCH_EVENTS)
bool screenHasTouchDevice()
{
    return true;
}

bool screenIsTouchPrimaryInputDevice()
{
    return true;
}
#endif

} // namespace WebCore
