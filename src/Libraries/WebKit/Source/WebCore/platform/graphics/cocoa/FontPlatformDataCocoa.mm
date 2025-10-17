/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
#import "config.h"
#import "FontPlatformData.h"

#import "FontCacheCoreText.h"
#import "SharedBuffer.h"
#import <pal/spi/cf/CoreTextSPI.h>
#import <wtf/Hasher.h>

#if PLATFORM(IOS_FAMILY)
#import <CoreText/CoreText.h>
#import <pal/spi/cg/CoreGraphicsSPI.h>
#endif

namespace WebCore {

unsigned FontPlatformData::hash() const
{
    // FIXME: Hashing a CFHash is unfortunate here.
    return computeHash(CFHash(m_font.get()), m_widthVariant, m_isHashTableDeletedValue, m_textRenderingMode, m_orientation, m_orientation, m_syntheticBold, m_syntheticOblique);
}

bool FontPlatformData::platformIsEqual(const FontPlatformData& other) const
{
    if (!m_font || !other.m_font)
        return m_font == other.m_font;
    return CFEqual(m_font.get(), other.m_font.get());
}

RefPtr<SharedBuffer> FontPlatformData::platformOpenTypeTable(uint32_t) const
{
    return nullptr;
}

Vector<FontPlatformData::FontVariationAxis> FontPlatformData::variationAxes(ShouldLocalizeAxisNames shouldLocalizeAxisNames) const
{
    auto platformFont = ctFont();
    if (!platformFont)
        return { };
    
    return WTF::map(defaultVariationValues(platformFont, shouldLocalizeAxisNames), [](auto&& entry) {
        auto& [tag, values] = entry;
        return FontPlatformData::FontVariationAxis { values.axisName, String(tag), values.defaultValue, values.minimumValue, values.maximumValue };
    });
}


} // namespace WebCore
