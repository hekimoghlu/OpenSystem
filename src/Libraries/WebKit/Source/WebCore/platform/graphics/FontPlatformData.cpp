/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
#include "FontPlatformData.h"


#include "FontCache.h"
#include "FontCustomPlatformData.h"
#include "FontDescription.h"
#include "RenderStyleConstants.h"
#include "StyleFontSizeFunctions.h"

#include <wtf/SortedArrayMap.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FontPlatformData);

FontPlatformData::FontPlatformData(WTF::HashTableDeletedValueType)
    : m_isHashTableDeletedValue(true)
{
}

FontPlatformData::FontPlatformData()
{
}

FontPlatformData::FontPlatformData(float size, bool syntheticBold, bool syntheticOblique, FontOrientation orientation, FontWidthVariant widthVariant, TextRenderingMode textRenderingMode, const FontCustomPlatformData* customPlatformData)
    : m_size(size)
    , m_orientation(orientation)
    , m_widthVariant(widthVariant)
    , m_textRenderingMode(textRenderingMode)
    , m_customPlatformData(customPlatformData)
    , m_syntheticBold(syntheticBold)
    , m_syntheticOblique(syntheticOblique)
{
}

FontPlatformData::~FontPlatformData() = default;
FontPlatformData::FontPlatformData(const FontPlatformData&) = default;
FontPlatformData& FontPlatformData::operator=(const FontPlatformData&) = default;

#if !USE(FREETYPE)
FontPlatformData FontPlatformData::cloneWithOrientation(const FontPlatformData& source, FontOrientation orientation)
{
    FontPlatformData copy(source);
    copy.m_orientation = orientation;
    return copy;
}

FontPlatformData FontPlatformData::cloneWithSyntheticOblique(const FontPlatformData& source, bool syntheticOblique)
{
    FontPlatformData copy(source);
    copy.m_syntheticOblique = syntheticOblique;
    return copy;
}
#endif

#if !USE(FREETYPE) && !PLATFORM(COCOA)
// FIXME: Don't other platforms also need to reinstantiate their copy.m_font for scaled size?
FontPlatformData FontPlatformData::cloneWithSize(const FontPlatformData& source, float size)
{
    FontPlatformData copy(source);
    copy.updateSize(size);
    return copy;
}

#if !USE(SKIA)
void FontPlatformData::updateSize(float size)
{
    m_size = size;
}
#endif
#endif

void FontPlatformData::updateSizeWithFontSizeAdjust(const FontSizeAdjust& fontSizeAdjust, float computedSize)
{
    if (!fontSizeAdjust.value)
        return;

    auto tmpFont = FontCache::forCurrentThread().fontForPlatformData(*this);
    auto adjustedFontSize = Style::adjustedFontSize(computedSize, fontSizeAdjust, tmpFont->fontMetrics());

    if (adjustedFontSize == size())
        return;

    updateSize(std::min(adjustedFontSize, maximumAllowedFontSize));
}

const FontPlatformData::CreationData* FontPlatformData::creationData() const
{
    return m_customPlatformData ? &m_customPlatformData->creationData : nullptr;
}

#if !PLATFORM(COCOA) && !USE(FREETYPE) && !USE(SKIA)
Vector<FontPlatformData::FontVariationAxis> FontPlatformData::variationAxes(ShouldLocalizeAxisNames) const
{
    // FIXME: <webkit.org/b/219614> Not implemented yet.
    return { };
}
#endif

}
