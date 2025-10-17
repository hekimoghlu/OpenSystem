/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
#include "PlatformMouseEvent.h"

#include "GDIUtilities.h"
#include "HWndDC.h"
#include <wtf/Assertions.h>
#include <wtf/MathExtras.h>
#include <windows.h>
#include <windowsx.h>

namespace WebCore {

#define HIGH_BIT_MASK_SHORT 0x8000

static IntPoint positionForEvent(HWND, LPARAM lParam)
{
    IntPoint point(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam));
    return point;
}

static IntPoint globalPositionForEvent(HWND hWnd, LPARAM lParam)
{
    POINT point = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
    ClientToScreen(hWnd, &point);
    return point;
}

static PlatformEvent::Type messageToEventType(UINT message)
{
    switch (message) {
        case WM_LBUTTONDBLCLK:
        case WM_RBUTTONDBLCLK:
        case WM_MBUTTONDBLCLK:
            //MSDN docs say double click is sent on mouse down
        case WM_LBUTTONDOWN:
        case WM_RBUTTONDOWN:
        case WM_MBUTTONDOWN:
            return PlatformEvent::Type::MousePressed;

        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP:
            return PlatformEvent::Type::MouseReleased;

        case WM_MOUSELEAVE:
        case WM_MOUSEMOVE:
            return PlatformEvent::Type::MouseMoved;

        default:
            ASSERT_NOT_REACHED();
            //Move is relatively harmless
            return PlatformEvent::Type::MouseMoved;
    }
}

PlatformMouseEvent::PlatformMouseEvent(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam, bool didActivateWebView)
    : PlatformEvent(messageToEventType(message), wParam & MK_SHIFT, wParam & MK_CONTROL, GetKeyState(VK_MENU) & HIGH_BIT_MASK_SHORT, GetKeyState(VK_MENU) & HIGH_BIT_MASK_SHORT, WallTime::now())
    , m_position(positionForEvent(hWnd, lParam))
    , m_globalPosition(globalPositionForEvent(hWnd, lParam))
    , m_clickCount(0)
    , m_modifierFlags(wParam)
    , m_buttons(buttonsForEvent(wParam))
    , m_didActivateWebView(didActivateWebView)
{
    switch (message) {
        case WM_LBUTTONDOWN:
        case WM_LBUTTONUP:
        case WM_LBUTTONDBLCLK:
            m_button = MouseButton::Left;
            break;
        case WM_RBUTTONDOWN:
        case WM_RBUTTONUP:
        case WM_RBUTTONDBLCLK:
            m_button = MouseButton::Right;
            break;
        case WM_MBUTTONDOWN:
        case WM_MBUTTONUP:
        case WM_MBUTTONDBLCLK:
            m_button = MouseButton::Middle;
            break;
        case WM_MOUSEMOVE:
        case WM_MOUSELEAVE:
            if (wParam & MK_LBUTTON)
                m_button = MouseButton::Left;
            else if (wParam & MK_MBUTTON)
                m_button = MouseButton::Middle;
            else if (wParam & MK_RBUTTON)
                m_button = MouseButton::Right;
            else
                m_button = MouseButton::None;
            break;
        default:
            ASSERT_NOT_REACHED();
    }
}

} // namespace WebCore
