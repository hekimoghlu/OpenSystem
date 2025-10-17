/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#pragma once

#include "WPEDisplayWayland.h"
#include <wayland-client.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/TZoneMalloc.h>

namespace WPE {

class WaylandCursorTheme;

class WaylandCursor {
    WTF_MAKE_TZONE_ALLOCATED(WaylandCursor);
public:
    explicit WaylandCursor(WPEDisplayWayland*);
    ~WaylandCursor();

    void setFromName(const char*, double);
    void setFromBuffer(struct wl_buffer*, uint32_t width, uint32_t height, uint32_t hotspotX, uint32_t hotspotY);
    void update() const;

private:
    WPEDisplayWayland* m_display { nullptr };
    struct wl_surface* m_surface { nullptr };
    std::unique_ptr<WaylandCursorTheme> m_theme;
    GUniquePtr<char> m_name;
    struct {
        int32_t x { 0 };
        int32_t y { 0 };
    } m_hotspot;
    mutable bool m_cursorChanged { false };
};

} // namespace WPE
