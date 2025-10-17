/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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

#include <optional>
#include <wayland-client.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/CString.h>

namespace WPE {

class CursorTheme;
class WaylandSHMPool;

class WaylandCursorTheme {
    WTF_MAKE_TZONE_ALLOCATED(WaylandCursorTheme);
public:
    static std::unique_ptr<WaylandCursorTheme> create(const char* path, uint32_t size, struct wl_shm*);
    static std::unique_ptr<WaylandCursorTheme> create(struct wl_shm*);

    WaylandCursorTheme(std::unique_ptr<CursorTheme>&&, std::unique_ptr<WaylandSHMPool>&&);
    ~WaylandCursorTheme();

    struct Image {
        uint32_t width { 0 };
        uint32_t height { 0 };
        uint32_t hotspotX { 0 };
        uint32_t hotspotY { 0 };
        struct wl_buffer* buffer { nullptr };
    };
    const Vector<Image>& cursor(const char*, double, std::optional<uint32_t> maxImages = std::nullopt);

private:
    void loadCursor(const char*, double, std::optional<uint32_t> maxImages, Vector<WaylandCursorTheme::Image>&);

    std::unique_ptr<CursorTheme> m_theme;
    std::unique_ptr<WaylandSHMPool> m_pool;
    HashMap<std::pair<CString, uint32_t>, Vector<Image>> m_cursors;
};

} // namespace WPE
