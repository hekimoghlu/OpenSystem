/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#include "PointerLockManager.h"

#include "Display.h"
#include "NativeWebMouseEvent.h"
#include "WebPageProxy.h"
#include <WebCore/PlatformMouseEvent.h>
#include <WebCore/PointerEvent.h>
#include <WebCore/PointerID.h>
#include <gtk/gtk.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(WAYLAND)
#include "PointerLockManagerWayland.h"
#endif

#if PLATFORM(X11)
#include "PointerLockManagerX11.h"
#endif

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PointerLockManager);

std::unique_ptr<PointerLockManager> PointerLockManager::create(WebPageProxy& webPage, const FloatPoint& position, const FloatPoint& globalPosition, WebMouseEventButton button, unsigned short buttons, OptionSet<WebEventModifier> modifiers)
{
#if PLATFORM(WAYLAND)
    if (Display::singleton().isWayland())
        return makeUnique<PointerLockManagerWayland>(webPage, position, globalPosition, button, buttons, modifiers);
#endif
#if PLATFORM(X11)
    if (Display::singleton().isX11())
        return makeUnique<PointerLockManagerX11>(webPage, position, globalPosition, button, buttons, modifiers);
#endif
    ASSERT_NOT_REACHED();
    return nullptr;
}

PointerLockManager::PointerLockManager(WebPageProxy& webPage, const FloatPoint& position, const FloatPoint& globalPosition, WebMouseEventButton button, unsigned short buttons, OptionSet<WebEventModifier> modifiers)
    : m_webPage(webPage)
    , m_position(position)
    , m_button(button)
    , m_buttons(buttons)
    , m_modifiers(modifiers)
    , m_initialPoint(globalPosition)
{
}

PointerLockManager::~PointerLockManager()
{
    ASSERT(!m_device);
}

bool PointerLockManager::lock()
{
    ASSERT(!m_device);

    m_device = gdk_seat_get_pointer(gdk_display_get_default_seat(gtk_widget_get_display(m_webPage.viewWidget())));
    return !!m_device;
}

bool PointerLockManager::unlock()
{
    if (!m_device)
        return false;

    m_device = nullptr;

    return true;
}

void PointerLockManager::handleMotion(const FloatSize& delta)
{
    m_webPage.handleMouseEvent(NativeWebMouseEvent(WebEventType::MouseMove, m_button, m_buttons, IntPoint(m_position), IntPoint(m_initialPoint), 0, m_modifiers, delta, mousePointerID, mousePointerEventType(), PlatformMouseEvent::IsTouch::No));
}

} // namespace WebKit
