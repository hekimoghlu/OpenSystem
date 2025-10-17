/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#ifndef OpenTypeVerticalData_h
#define OpenTypeVerticalData_h

#if ENABLE(OPENTYPE_VERTICAL)

#include "Glyph.h"
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class Font;
class FontPlatformData;
class GlyphPage;

class OpenTypeVerticalData : public RefCounted<OpenTypeVerticalData> {
public:
    static RefPtr<OpenTypeVerticalData> create(const FontPlatformData&);

    bool hasVerticalMetrics() const { return !m_advanceHeights.isEmpty(); }
    float advanceHeight(const Font*, Glyph) const;
    void getVerticalTranslationsForGlyphs(const Font*, const Glyph*, size_t, float* outXYArray) const;
    void substituteWithVerticalGlyphs(const Font*, GlyphPage*) const;

private:
    explicit OpenTypeVerticalData(const FontPlatformData&, Vector<uint16_t>&& advanceWidths);

    void loadMetrics(const FontPlatformData&);
    void loadVerticalGlyphSubstitutions(const FontPlatformData&);
    bool hasVORG() const { return !m_vertOriginY.isEmpty(); }

    UncheckedKeyHashMap<Glyph, Glyph> m_verticalGlyphMap;
    Vector<uint16_t> m_advanceWidths;
    Vector<uint16_t> m_advanceHeights;
    Vector<int16_t> m_topSideBearings;
    int16_t m_defaultVertOriginY { 0 };
    UncheckedKeyHashMap<Glyph, int16_t> m_vertOriginY;
};

} // namespace WebCore

#endif // ENABLE(OPENTYPE_VERTICAL)

#endif // OpenTypeVerticalData_h
