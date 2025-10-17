/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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

#if USE(SKIA)

#include "HbUniquePtr.h"
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkFont.h>
#include <skia/core/SkTypeface.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#include <wtf/RefCounted.h>

namespace WebCore {

class FontPlatformData;

class SkiaHarfBuzzFont final : public RefCounted<SkiaHarfBuzzFont> {
public:
    static Ref<SkiaHarfBuzzFont> getOrCreate(SkTypeface&);

    hb_font_t* scaledFont(const FontPlatformData&);

    std::optional<hb_codepoint_t> glyph(hb_codepoint_t, std::optional<hb_codepoint_t> variation = std::nullopt);
    hb_position_t glyphWidth(hb_codepoint_t);
    void glyphWidths(unsigned count, const hb_codepoint_t* glyphs, unsigned glyphStride, hb_position_t* advances, unsigned advanceStride);
    void glyphExtents(hb_codepoint_t, hb_glyph_extents_t*);

    ~SkiaHarfBuzzFont();

private:
    friend class SkiaHarfBuzzFontCache;

    explicit SkiaHarfBuzzFont(SkTypeface&);

    SkTypefaceID m_uniqueID { 0 };
    HbUniquePtr<hb_font_t> m_font;
    SkFont m_scaledFont;
};

} // namespace WebCore

#endif // USE(SKIA)
