/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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
#include "WPEWaylandCursor.h"

#include "WPEDisplayWaylandPrivate.h"
#include "WPEWaylandCursorTheme.h"
#include <wtf/TZoneMallocInlines.h>

namespace WPE {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WaylandCursor);

WaylandCursor::WaylandCursor(WPEDisplayWayland* display)
    : m_display(display)
    , m_surface(wl_compositor_create_surface(wpe_display_wayland_get_wl_compositor(m_display)))
    , m_theme(WaylandCursorTheme::create(wpe_display_wayland_get_wl_shm(m_display)))
{
    if (!m_theme)
        g_warning("Could not load cursor theme, disabling named cursors support");
}

WaylandCursor::~WaylandCursor()
{
    if (m_surface)
        wl_surface_destroy(m_surface);
}

void WaylandCursor::setFromName(const char* name, double scale)
{
    if (!m_theme)
        return;

    if (!g_strcmp0(m_name.get(), name))
        return;

    m_name.reset(g_strdup(name));
    if (!g_strcmp0(m_name.get(), "none")) {
        m_cursorChanged = true;
        update();
        wl_surface_attach(m_surface, nullptr, 0, 0);
        wl_surface_commit(m_surface);
        return;
    }

    // FIXME: support animated cursors.
    const auto& cursor = m_theme->cursor(name, scale, 1);
    if (cursor.isEmpty()) {
        g_warning("Cursor %s not found in theme", name);
        return;
    }

    m_hotspot.x = cursor[0].hotspotX;
    m_hotspot.y = cursor[0].hotspotY;
    m_cursorChanged = true;
    update();

    wl_surface_attach(m_surface, cursor[0].buffer, 0, 0);
    if (wl_surface_get_version(m_surface) >= WL_SURFACE_SET_BUFFER_SCALE_SINCE_VERSION)
        wl_surface_set_buffer_scale(m_surface, scale);
    wl_surface_damage(m_surface, 0, 0, cursor[0].width, cursor[0].height);
    wl_surface_commit(m_surface);
}

void WaylandCursor::setFromBuffer(struct wl_buffer* buffer, uint32_t width, uint32_t height, uint32_t hotspotX, uint32_t hotspotY)
{
    m_name = nullptr;
    m_hotspot.x = hotspotX;
    m_hotspot.y = hotspotY;
    m_cursorChanged = true;
    update();

    wl_surface_attach(m_surface, buffer, 0, 0);
    wl_surface_damage(m_surface, 0, 0, width, height);
    wl_surface_commit(m_surface);
}

void WaylandCursor::update() const
{
    if (!m_cursorChanged)
        return;
    if (auto* seat = wpeDisplayWaylandGetSeat(m_display))
        seat->setCursor(m_surface, m_hotspot.x, m_hotspot.y);
    m_cursorChanged = false;
}

} // namespace WPE
