/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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
#include "PlatformWheelEvent.h"

#include "FloatPoint.h"
#include "FloatSize.h"
#include "GDIUtilities.h"
#include "HWndDC.h"
#include <windows.h>
#include <windowsx.h>

namespace WebCore {

#define HIGH_BIT_MASK_SHORT 0x8000
#define SPI_GETWHEELSCROLLCHARS 0x006C

static IntPoint positionForEvent(HWND hWnd, LPARAM lParam)
{
    POINT point = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
    ScreenToClient(hWnd, &point);
    IntPoint logicalPoint(point);
    float inverseScaleFactor = 1.0f / deviceScaleFactorForWindow(hWnd);
    logicalPoint.scale(inverseScaleFactor, inverseScaleFactor);
    return logicalPoint;
}

static IntPoint globalPositionForEvent(HWND hWnd, LPARAM lParam)
{
    IntPoint logicalPoint(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
    float inverseScaleFactor = 1.0f / deviceScaleFactorForWindow(hWnd);
    logicalPoint.scale(inverseScaleFactor, inverseScaleFactor);
    return logicalPoint;
}

static int horizontalScrollChars()
{
    static ULONG scrollChars;
    if (!scrollChars && !SystemParametersInfo(SPI_GETWHEELSCROLLCHARS, 0, &scrollChars, 0))
        scrollChars = 1;
    return scrollChars;
}

static unsigned verticalScrollLines()
{
    static ULONG scrollLines;
    if (!scrollLines && !SystemParametersInfo(SPI_GETWHEELSCROLLLINES, 0, &scrollLines, 0))
        scrollLines = 3;
    return scrollLines;
}

PlatformWheelEvent::PlatformWheelEvent(HWND hWnd, WPARAM wParam, LPARAM lParam, bool isMouseHWheel)
    : PlatformEvent(PlatformEvent::Type::Wheel, wParam & MK_SHIFT, wParam & MK_CONTROL, GetKeyState(VK_MENU) & HIGH_BIT_MASK_SHORT, GetKeyState(VK_MENU) & HIGH_BIT_MASK_SHORT, WallTime::fromRawSeconds(::GetTickCount() * 0.001))
    , m_position(positionForEvent(hWnd, lParam))
    , m_globalPosition(globalPositionForEvent(hWnd, lParam))
{
    float scaleFactor = deviceScaleFactorForWindow(hWnd);

    float delta = GET_WHEEL_DELTA_WPARAM(wParam) / (scaleFactor * static_cast<float>(WHEEL_DELTA));
    if (isMouseHWheel) {
        // Windows is <-- -/+ -->, WebKit wants <-- +/- -->, so we negate
        // |delta| after saving the original value on the wheel tick member.
        m_wheelTicksX = delta;
        m_wheelTicksY = 0;
        delta = -delta;
    } else {
        // Even though we use shift + vertical wheel to scroll horizontally in
        // WebKit, we still note it as a vertical scroll on the wheel tick
        // member, so that the DOM event we later construct will match the real
        // hardware event better.
        m_wheelTicksX = 0;
        m_wheelTicksY = delta;
    }
    if (isMouseHWheel || shiftKey()) {
        m_deltaX = delta * static_cast<float>(horizontalScrollChars()) * cScrollbarPixelsPerLine;
        m_deltaY = 0;
        m_granularity = ScrollByPixelWheelEvent;
    } else {
        m_deltaX = 0;
        m_deltaY = delta;
        unsigned verticalMultiplier = verticalScrollLines();
        m_granularity = (verticalMultiplier == WHEEL_PAGESCROLL) ? ScrollByPageWheelEvent : ScrollByPixelWheelEvent;
        if (m_granularity == ScrollByPixelWheelEvent)
            m_deltaY *= static_cast<float>(verticalMultiplier) * cScrollbarPixelsPerLine;
    }
}

}
