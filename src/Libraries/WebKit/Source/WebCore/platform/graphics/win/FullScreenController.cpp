/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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

#if ENABLE(FULLSCREEN_API)

#include "FullScreenController.h"

#include "Element.h"
#include "FullScreenControllerClient.h"
#include "FullScreenWindow.h"
#include "IntRect.h"
#include "Timer.h"
#include "WebCoreInstanceHandle.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static const int kFullScreenAnimationDuration = 500; // milliseconds 

class FullScreenController::Private : public FullScreenClient  {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(Private);
public:
    Private(FullScreenController* controller, FullScreenControllerClient* client) 
        : m_controller(controller)
        , m_client(client)
        , m_originalHost(0)
        , m_isFullScreen(false)
        , m_isEnteringFullScreen(false)
        , m_isExitingFullScreen(false)
    {
    }
    virtual ~Private() = default;

    virtual LRESULT fullscreenClientWndProc(HWND, UINT, WPARAM, LPARAM);
    
    FullScreenController* m_controller;
    FullScreenControllerClient* m_client;
    std::unique_ptr<FullScreenWindow> m_fullScreenWindow;
    std::unique_ptr<FullScreenWindow> m_backgroundWindow;
    IntRect m_fullScreenFrame;
    IntRect m_originalFrame;
    HWND m_originalHost;
    bool m_isFullScreen;
    bool m_isEnteringFullScreen;
    bool m_isExitingFullScreen;
};

LRESULT FullScreenController::Private::fullscreenClientWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    LRESULT lResult = 0;

    switch (msg) {
    case WM_MOVE:
        m_fullScreenFrame.setX(LOWORD(lParam));
        m_fullScreenFrame.setY(HIWORD(lParam));
        break;
    case WM_SIZE:
        m_fullScreenFrame.setWidth(LOWORD(lParam));
        m_fullScreenFrame.setHeight(HIWORD(lParam));
        if (m_client->fullScreenClientWindow())
            ::SetWindowPos(m_client->fullScreenClientWindow(), 0, 0, 0, m_fullScreenFrame.width(), m_fullScreenFrame.height(), SWP_NOREPOSITION  | SWP_NOMOVE);
        break;
    case WM_ACTIVATE:
        // Because m_fullScreenWindow is a topmost window, we need to exit full screen explicitly when it's deactivated.
        if (!wParam && m_fullScreenWindow && (hwnd == m_fullScreenWindow->hwnd()))
            m_controller->exitFullScreen();
        break;
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) {
            m_controller->exitFullScreen();
            break;
        }
        FALLTHROUGH;
    default:
        lResult = ::DefWindowProc(hwnd, msg, wParam, lParam);
    }

    return lResult;
}

FullScreenController::FullScreenController(FullScreenControllerClient* client)
    : m_private(makeUnique<Private>(this, client))
{
    ASSERT_ARG(client, client);
}

FullScreenController::~FullScreenController() = default;

bool FullScreenController::isFullScreen() const
{
    return m_private->m_isFullScreen;
}

void FullScreenController::enterFullScreen()
{
    if (m_private->m_isFullScreen || m_private->m_isEnteringFullScreen)
        return;
    m_private->m_isFullScreen = true;
    m_private->m_isEnteringFullScreen = true;

    m_private->m_client->fullScreenClientSaveScrollPosition();

    m_private->m_originalHost = m_private->m_client->fullScreenClientParentWindow();
    RECT originalFrame = {0, 0, 0, 0};
    ::GetClientRect(m_private->m_client->fullScreenClientWindow(), &originalFrame);
    ::MapWindowPoints(m_private->m_client->fullScreenClientWindow(), m_private->m_originalHost, reinterpret_cast<LPPOINT>(&originalFrame), 2);
    m_private->m_originalFrame = originalFrame;

    ASSERT(!m_private->m_backgroundWindow);
    m_private->m_backgroundWindow = makeUnique<FullScreenWindow>(m_private.get());
    m_private->m_backgroundWindow->createWindow(0);
    ::AnimateWindow(m_private->m_backgroundWindow->hwnd(), kFullScreenAnimationDuration, AW_BLEND | AW_ACTIVATE);

    m_private->m_client->fullScreenClientWillEnterFullScreen();
    ASSERT(!m_private->m_fullScreenWindow);
    m_private->m_fullScreenWindow = makeUnique<FullScreenWindow>(m_private.get());
    ASSERT(m_private->m_fullScreenWindow);
    m_private->m_fullScreenWindow->createWindow(0);

    m_private->m_client->fullScreenClientSetParentWindow(m_private->m_fullScreenWindow->hwnd());

    IntRect viewFrame(IntPoint(), m_private->m_fullScreenFrame.size());
    ::SetWindowPos(m_private->m_fullScreenWindow->hwnd(), HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    ::SetWindowPos(m_private->m_client->fullScreenClientWindow(), HWND_TOP, 0, 0, viewFrame.width(), viewFrame.height(), SWP_NOACTIVATE);

    m_private->m_client->fullScreenClientDidEnterFullScreen();
    m_private->m_client->fullScreenClientForceRepaint();
}

void FullScreenController::enterFullScreenRepaintCompleted()
{
    if (!m_private->m_isEnteringFullScreen)
        return;
    m_private->m_isEnteringFullScreen = false;

    // Normally, when the background fullscreen window is animated in, the Windows taskbar will be hidden, but this doesn't always work for some reason.
    // Setting the real fullscreen window to be a topmost window will force the taskbar to be hidden when we call AnimateWindow() below if it wasn't before.
    ::SetWindowPos(m_private->m_fullScreenWindow->hwnd(), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    ::AnimateWindow(m_private->m_fullScreenWindow->hwnd(), kFullScreenAnimationDuration, AW_BLEND | AW_ACTIVATE);
}

void FullScreenController::exitFullScreen()
{
    if (!m_private->m_isFullScreen || m_private->m_isExitingFullScreen)
        return;
    m_private->m_isFullScreen = false;
    m_private->m_isExitingFullScreen = true;

    ::AnimateWindow(m_private->m_fullScreenWindow->hwnd(), kFullScreenAnimationDuration, AW_HIDE | AW_BLEND);

    m_private->m_client->fullScreenClientWillExitFullScreen();
    m_private->m_client->fullScreenClientSetParentWindow(m_private->m_originalHost);
    m_private->m_fullScreenWindow = nullptr;

    ::SetWindowPos(m_private->m_client->fullScreenClientWindow(), 0, m_private->m_originalFrame.x(), m_private->m_originalFrame.y(), m_private->m_originalFrame.width(), m_private->m_originalFrame.height(), SWP_NOACTIVATE | SWP_NOZORDER);

    m_private->m_client->fullScreenClientRestoreScrollPosition();
    m_private->m_client->fullScreenClientDidExitFullScreen();
    m_private->m_client->fullScreenClientForceRepaint();
}

void FullScreenController::exitFullScreenRepaintCompleted()
{
    if (!m_private->m_isExitingFullScreen)
        return;
    m_private->m_isExitingFullScreen = false;

    ASSERT(m_private->m_backgroundWindow);
    ::AnimateWindow(m_private->m_backgroundWindow->hwnd(), kFullScreenAnimationDuration, AW_HIDE | AW_BLEND);
    m_private->m_backgroundWindow = nullptr;
}

void FullScreenController::repaintCompleted()
{
    if (m_private->m_isEnteringFullScreen)
        enterFullScreenRepaintCompleted();
    else if (m_private->m_isExitingFullScreen)
        exitFullScreenRepaintCompleted();
}

void FullScreenController::close()
{
    if (!m_private->m_isFullScreen)
        return;
    m_private->m_isFullScreen = false;

    m_private->m_client->fullScreenClientWillExitFullScreen();
    m_private->m_client->fullScreenClientSetParentWindow(m_private->m_originalHost);
    m_private->m_fullScreenWindow = nullptr;

    m_private->m_client->fullScreenClientDidExitFullScreen();
    ::SetWindowPos(m_private->m_client->fullScreenClientWindow(), 0, m_private->m_originalFrame.x(), m_private->m_originalFrame.y(), m_private->m_originalFrame.width(), m_private->m_originalFrame.height(), SWP_NOACTIVATE | SWP_NOZORDER);
    ::RedrawWindow(m_private->m_client->fullScreenClientWindow(), 0, 0, RDW_INVALIDATE | RDW_UPDATENOW | RDW_ERASE | RDW_ALLCHILDREN);
    m_private->m_backgroundWindow = nullptr;
}

} // namespace WebCore

#endif
