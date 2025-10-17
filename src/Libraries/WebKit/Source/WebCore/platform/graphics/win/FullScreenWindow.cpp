/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#include "FullScreenWindow.h"

#if ENABLE(FULLSCREEN_API)

#include "IntRect.h"
#include "WebCoreInstanceHandle.h"
#include <windows.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FullScreenWindow);

FullScreenWindow::FullScreenWindow(FullScreenClient* client)
    : m_client(client)
    , m_hwnd(0)
{
}

FullScreenWindow::~FullScreenWindow()
{
    if (!m_hwnd)
        return;

    ::DestroyWindow(m_hwnd);
    ASSERT(!m_hwnd);
}

void FullScreenWindow::createWindow(HWND parentHwnd)
{
    static ATOM windowAtom;
    static LPCWSTR windowClassName = L"FullscreenWindowClass";
    if (!windowAtom) {
        WNDCLASSEX wcex { };
        wcex.cbSize = sizeof(wcex);
        wcex.style = CS_HREDRAW | CS_VREDRAW;
        wcex.lpfnWndProc = staticWndProc;
        wcex.hInstance = instanceHandle();
        wcex.lpszClassName = windowClassName;
        windowAtom = ::RegisterClassEx(&wcex);
    }

    ASSERT(!m_hwnd);

    MONITORINFO mi { };
    mi.cbSize = sizeof(MONITORINFO);
    if (!GetMonitorInfo(MonitorFromWindow(parentHwnd, MONITOR_DEFAULTTONEAREST), &mi))
        return;

    IntRect monitorRect = mi.rcMonitor;
    
    ::CreateWindowExW(WS_EX_TOOLWINDOW, windowClassName, L"", WS_POPUP, 
        monitorRect.x(), monitorRect.y(), monitorRect.width(), monitorRect.height(),
        parentHwnd, 0, instanceHandle(), this);
    ASSERT(IsWindow(m_hwnd));

    ::SetFocus(m_hwnd);
}

LRESULT FullScreenWindow::staticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    LONG_PTR longPtr = GetWindowLongPtr(hWnd, GWLP_USERDATA);

    if (!longPtr && message == WM_CREATE) {
        LPCREATESTRUCT lpcs = reinterpret_cast<LPCREATESTRUCT>(lParam);
        longPtr = reinterpret_cast<LONG_PTR>(lpcs->lpCreateParams);
        ::SetWindowLongPtr(hWnd, GWLP_USERDATA, longPtr);
    }

    if (FullScreenWindow* window = reinterpret_cast<FullScreenWindow*>(longPtr))
        return window->wndProc(hWnd, message, wParam, lParam);

    return ::DefWindowProc(hWnd, message, wParam, lParam);    
}

LRESULT FullScreenWindow::wndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    LRESULT lResult = 0;
    switch (message) {
    case WM_CREATE:
        m_hwnd = hWnd;
        break;
    case WM_DESTROY:
        m_hwnd = 0;
        break;
    case WM_WINDOWPOSCHANGED:
        {
            LPWINDOWPOS wp = reinterpret_cast<LPWINDOWPOS>(lParam);
            if (wp->flags & SWP_NOSIZE)
                break;
        }
        break;
    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = ::BeginPaint(m_hwnd, &ps);
            ::FillRect(hdc, &ps.rcPaint, (HBRUSH)::GetStockObject(BLACK_BRUSH));
            ::EndPaint(m_hwnd, &ps);
        }
        break;
    case WM_PRINTCLIENT:
        {
            RECT clientRect;
            HDC context = (HDC)wParam;
            ::GetClientRect(m_hwnd, &clientRect);
            ::FillRect(context, &clientRect, (HBRUSH)::GetStockObject(BLACK_BRUSH));
        }
    }
    if (m_client)
        lResult = m_client->fullscreenClientWndProc(hWnd, message, wParam, lParam);

    return lResult;
}

} // namespace WebCore

#endif // ENABLE(FULLSCREEN_API)
