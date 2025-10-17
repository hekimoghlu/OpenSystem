/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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

#include <gbm.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GUniquePtr.h>

namespace WPE {

namespace DRM {

class Buffer;
class CursorTheme;
class Plane;

class Cursor {
    WTF_MAKE_TZONE_ALLOCATED(Cursor);
public:
    Cursor(std::unique_ptr<Plane>&&, struct gbm_device*, uint32_t cursorWidth, uint32_t cursorHeight);
    ~Cursor();

    void setFromName(const char*, double);
    void setFromBytes(GBytes*, uint32_t width, uint32_t height, uint32_t stride, uint32_t hotspotX, uint32_t hotspotY);
    bool setPosition(uint32_t x, uint32_t y);
    uint32_t x() const { return m_position.x - m_hotspot.x; }
    uint32_t y() const { return m_position.y - m_hotspot.y; }
    uint32_t width() const { return m_deviceWidth; }
    uint32_t height() const { return m_deviceHeight; }

    const Plane& plane() const { return *m_plane; }
    Buffer* buffer() const { return m_isHidden ? nullptr : m_buffer.get(); }

private:
    bool tryEnsureBuffer();
    void updateBuffer(const uint8_t*, uint32_t width, uint32_t height, uint32_t stride);

    std::unique_ptr<Plane> m_plane;
    struct gbm_device* m_device { nullptr };
    uint32_t m_deviceWidth { 0 };
    uint32_t m_deviceHeight { 0 };
    std::unique_ptr<CursorTheme> m_theme;
    bool m_isHidden { false };
    GUniquePtr<char> m_name;
    std::unique_ptr<Buffer> m_buffer;
    struct {
        uint32_t x { 0 };
        uint32_t y { 0 };
    } m_position;
    struct {
        uint32_t x { 0 };
        uint32_t y { 0 };
    } m_hotspot;
};

} // namespace DRM

} // namespace WPE
