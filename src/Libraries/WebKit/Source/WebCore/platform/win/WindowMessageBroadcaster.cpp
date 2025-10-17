/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#include "WindowMessageBroadcaster.h"

#include "WindowMessageListener.h"

namespace WebCore {

typedef UncheckedKeyHashMap<HWND, WindowMessageBroadcaster*> InstanceMap;

static InstanceMap& instancesMap()
{
    static InstanceMap instances;
    return instances;
}

void WindowMessageBroadcaster::addListener(HWND hwnd, WindowMessageListener* listener)
{
    WindowMessageBroadcaster* broadcaster = instancesMap().get(hwnd);
    if (!broadcaster) {
        broadcaster = new WindowMessageBroadcaster(hwnd);
        instancesMap().add(hwnd, broadcaster);
    }

    broadcaster->addListener(listener);
}

void WindowMessageBroadcaster::removeListener(HWND hwnd, WindowMessageListener* listener)
{
    WindowMessageBroadcaster* broadcaster = instancesMap().get(hwnd);
    if (!broadcaster)
        return;

    broadcaster->removeListener(listener);
}

WindowMessageBroadcaster::WindowMessageBroadcaster(HWND hwnd)
    : m_subclassedWindow(hwnd)
    , m_originalWndProc(0)
{
    ASSERT_ARG(hwnd, IsWindow(hwnd));
}

WindowMessageBroadcaster::~WindowMessageBroadcaster() = default;

void WindowMessageBroadcaster::addListener(WindowMessageListener* listener)
{
    if (m_listeners.isEmpty()) {
        ASSERT(!m_originalWndProc);
#pragma warning(disable: 4244 4312)
        m_originalWndProc = reinterpret_cast<WNDPROC>(SetWindowLongPtr(m_subclassedWindow, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(SubclassedWndProc)));
    }
    ASSERT(m_originalWndProc);

    m_listeners.add(listener);
}

void WindowMessageBroadcaster::removeListener(WindowMessageListener* listener)
{
    m_listeners.remove(listener);
    if (m_listeners.isEmpty())
        destroy();
}

void WindowMessageBroadcaster::destroy()
{
    m_listeners.clear();
    unsubclassWindow();
    instancesMap().remove(m_subclassedWindow);
    delete this;
}

void WindowMessageBroadcaster::unsubclassWindow()
{
    SetWindowLongPtr(m_subclassedWindow, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(m_originalWndProc));
    m_originalWndProc = 0;
}

LRESULT CALLBACK WindowMessageBroadcaster::SubclassedWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    WindowMessageBroadcaster* broadcaster = instancesMap().get(hwnd);
    ASSERT(broadcaster);
    if (!broadcaster)
        return 0;

    ListenerSet::const_iterator end = broadcaster->listeners().end();
    for (ListenerSet::const_iterator it = broadcaster->listeners().begin(); it != end; ++it)
        (*it)->windowReceivedMessage(hwnd, message, wParam, lParam);

    WNDPROC originalWndProc = broadcaster->originalWndProc();

    // This will delete broadcaster.
    if (message == WM_DESTROY)
        broadcaster->destroy();

    return CallWindowProc(originalWndProc, hwnd, message, wParam, lParam);
}

} // namespace WebCore
