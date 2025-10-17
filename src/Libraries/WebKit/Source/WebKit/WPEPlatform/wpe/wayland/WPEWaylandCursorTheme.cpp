/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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
#include "WPEWaylandCursorTheme.h"

#include "WPECursorTheme.h"
#include "WPEWaylandSHMPool.h"
#include <wtf/TZoneMallocInlines.h>

namespace WPE {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WaylandCursorTheme);

std::unique_ptr<WaylandCursorTheme> WaylandCursorTheme::create(const char* name, uint32_t size, struct wl_shm* shm)
{
    auto theme = CursorTheme::create(name, size);
    if (!theme)
        return nullptr;

    auto pool = WaylandSHMPool::create(shm, size * size * 4);
    if (!pool)
        return nullptr;

    return makeUnique<WaylandCursorTheme>(WTFMove(theme), WTFMove(pool));
}

std::unique_ptr<WaylandCursorTheme> WaylandCursorTheme::create(struct wl_shm* shm)
{
    auto theme = CursorTheme::create();
    if (!theme)
        return nullptr;

    auto pool = WaylandSHMPool::create(shm, theme->size() * theme->size() * 4);
    if (!pool)
        return nullptr;

    return makeUnique<WaylandCursorTheme>(WTFMove(theme), WTFMove(pool));
}

WaylandCursorTheme::WaylandCursorTheme(std::unique_ptr<CursorTheme>&& theme, std::unique_ptr<WaylandSHMPool>&& pool)
    : m_theme(WTFMove(theme))
    , m_pool(WTFMove(pool))
{

}

WaylandCursorTheme::~WaylandCursorTheme()
{
}

const Vector<WaylandCursorTheme::Image>& WaylandCursorTheme::cursor(const char* name, double scale, std::optional<uint32_t> maxImages)
{
    uint32_t size = m_theme->size() * static_cast<uint32_t>(scale);
    auto addResult = m_cursors.add({ CString(name), size }, Vector<Image> { });
    if (addResult.isNewEntry)
        loadCursor(name, scale, maxImages, addResult.iterator->value);
    return addResult.iterator->value;
}

void WaylandCursorTheme::loadCursor(const char* name, double scale, std::optional<uint32_t> maxImages, Vector<WaylandCursorTheme::Image>& images)
{
    // Try first with the scaled size.
    uint32_t scaledSize = m_theme->size() * static_cast<uint32_t>(scale);
    auto cursor = m_theme->loadCursor(name, scaledSize, maxImages);
    if (cursor.isEmpty())
        return;

    int effectiveScale = 1;
    if (cursor[0].width != scaledSize || cursor[0].height != scaledSize) {
        // Scaled size not found, use the original size.
        cursor = m_theme->loadCursor(name, m_theme->size(), maxImages);
        effectiveScale = scale;
        if (cursor.isEmpty())
            return;
    }

    for (const auto& cursorImage : cursor) {
        Image image = { cursorImage.width * effectiveScale, cursorImage.height * effectiveScale, cursorImage.hotspotX * effectiveScale, cursorImage.hotspotY * effectiveScale, nullptr };
        auto sizeInBytes = image.width * image.height * 4;
        int offset = m_pool->allocate(sizeInBytes);
        if (offset < 0)
            return;

        // FIXME: support upscaling cursor.
        if (effectiveScale == 1)
            memcpy(reinterpret_cast<char*>(m_pool->data()) + offset, cursorImage.pixels.data(), sizeInBytes);

        image.buffer = m_pool->createBuffer(offset, image.width, image.height, image.width * 4);
        images.append(WTFMove(image));
    }
}

} // namespace WPE
