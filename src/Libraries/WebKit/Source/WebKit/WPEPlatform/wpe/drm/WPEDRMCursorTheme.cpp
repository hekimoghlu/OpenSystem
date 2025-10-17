/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
#include "WPEDRMCursorTheme.h"

#include "WPECursorTheme.h"
#include "WPEDRM.h"
#include <wtf/TZoneMallocInlines.h>

namespace WPE {

namespace DRM {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CursorTheme);

std::unique_ptr<CursorTheme> CursorTheme::create(const char* name, uint32_t size)
{
    auto theme = WPE::CursorTheme::create(name, size);
    if (!theme)
        return nullptr;

    return makeUnique<CursorTheme>(WTFMove(theme));
}

std::unique_ptr<CursorTheme> CursorTheme::create()
{
    auto theme = WPE::CursorTheme::create();
    if (!theme)
        return nullptr;

    return makeUnique<CursorTheme>(WTFMove(theme));
}

CursorTheme::CursorTheme(std::unique_ptr<WPE::CursorTheme>&& theme)
    : m_theme(WTFMove(theme))
{
}

CursorTheme::~CursorTheme()
{
}

const Vector<CursorTheme::Image>& CursorTheme::cursor(const char* name, double scale, std::optional<uint32_t> maxImages)
{
    uint32_t size = m_theme->size() * static_cast<uint32_t>(scale);
    auto addResult = m_cursors.add({ CString(name), size }, Vector<Image> { });
    if (addResult.isNewEntry)
        loadCursor(name, scale, maxImages, addResult.iterator->value);
    return addResult.iterator->value;
}

void CursorTheme::loadCursor(const char* name, double scale, std::optional<uint32_t> maxImages, Vector<CursorTheme::Image>& images)
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

    for (auto& cursorImage : cursor) {
        images.append({ cursorImage.width * effectiveScale, cursorImage.height * effectiveScale,
            cursorImage.hotspotX * effectiveScale, cursorImage.hotspotY * effectiveScale, WTFMove(cursorImage.pixels) });
    }
}

} // namespace DRM

} // namespace WPE
