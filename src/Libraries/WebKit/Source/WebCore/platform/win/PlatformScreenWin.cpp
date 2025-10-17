/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#include "IntRect.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "NotImplemented.h"
#include <windows.h>

namespace WebCore {

// Returns info for the default monitor if widget is NULL
static MONITORINFOEX monitorInfoForWidget(Widget* widget)
{
    HWND window = widget ? widget->root()->hostWindow()->platformPageClient() : 0;
    HMONITOR monitor = MonitorFromWindow(window, MONITOR_DEFAULTTOPRIMARY);

    MONITORINFOEX monitorInfo;
    monitorInfo.cbSize = sizeof(MONITORINFOEX);
    GetMonitorInfo(monitor, &monitorInfo);
    return monitorInfo;
}

static DEVMODE deviceInfoForWidget(Widget* widget)
{
    DEVMODE deviceInfo;
    deviceInfo.dmSize = sizeof(DEVMODE);
    deviceInfo.dmDriverExtra = 0;
    MONITORINFOEX monitorInfo = monitorInfoForWidget(widget);
    EnumDisplaySettings(monitorInfo.szDevice, ENUM_CURRENT_SETTINGS, &deviceInfo);

    return deviceInfo;
}

int screenDepth(Widget* widget)
{
    DEVMODE deviceInfo = deviceInfoForWidget(widget);
    if (deviceInfo.dmBitsPerPel == 32) {
        // Some video drivers return 32, but this function is supposed to ignore the alpha
        // component. See <http://webkit.org/b/42972>.
        return 24;
    }
    return deviceInfo.dmBitsPerPel;
}

int screenDepthPerComponent(Widget* widget)
{
    // FIXME: Assumes RGB -- not sure if this is right.
    return screenDepth(widget) / 3;
}

bool screenIsMonochrome(Widget* widget)
{
    DEVMODE deviceInfo = deviceInfoForWidget(widget);
    return deviceInfo.dmColor == DMCOLOR_MONOCHROME;
}

bool screenHasInvertedColors()
{
    return false;
}

FloatRect screenRect(Widget* widget)
{
    MONITORINFOEX monitorInfo = monitorInfoForWidget(widget);
    return monitorInfo.rcMonitor;
}

FloatRect screenAvailableRect(Widget* widget)
{
    MONITORINFOEX monitorInfo = monitorInfoForWidget(widget);
    return monitorInfo.rcWork;
}

DestinationColorSpace screenColorSpace(Widget*)
{
    return DestinationColorSpace::SRGB();
}

bool screenSupportsExtendedColor(Widget*)
{
    return false;
}

} // namespace WebCore
