/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
#include "FontFamilySpecificationCoreText.h"

#include "FontCache.h"
#include "FontFamilySpecificationCoreTextCache.h"
#include "FontSelector.h"
#include "StyleFontSizeFunctions.h"
#include "UnrealizedCoreTextFont.h"
#include <pal/spi/cf/CoreTextSPI.h>
#include <wtf/HashFunctions.h>
#include <wtf/HashMap.h>

#include <CoreText/CoreText.h>

namespace WebCore {

FontFamilySpecificationCoreText::FontFamilySpecificationCoreText(CTFontDescriptorRef fontDescriptor)
    : m_fontDescriptor(fontDescriptor)
{
}

FontFamilySpecificationCoreText::~FontFamilySpecificationCoreText() = default;

FontRanges FontFamilySpecificationCoreText::fontRanges(const FontDescription& fontDescription) const
{
    auto size = fontDescription.computedSize();
    auto& originalPlatformData = FontFamilySpecificationCoreTextCache::forCurrentThread().ensure(FontFamilySpecificationKey(m_fontDescriptor.get(), fontDescription), [&]() {
        // FIXME: Stop creating this unnecessary CTFont once rdar://problem/105508842 is fixed.
        UnrealizedCoreTextFont unrealizedFont = { adoptCF(CTFontCreateWithFontDescriptor(m_fontDescriptor.get(), size, nullptr)) };
        unrealizedFont.setSize(size);

        auto font = preparePlatformFont(WTFMove(unrealizedFont), fontDescription, { }, FontTypeForPreparation::SystemFont);

        auto [syntheticBold, syntheticOblique] = computeNecessarySynthesis(font.get(), fontDescription, { }, ShouldComputePhysicalTraits::Yes).boldObliquePair();

        auto platformData = makeUnique<FontPlatformData>(font.get(), size, false, syntheticOblique, fontDescription.orientation(), fontDescription.widthVariant(), fontDescription.textRenderingMode());
        platformData->updateSizeWithFontSizeAdjust(fontDescription.fontSizeAdjust(), fontDescription.computedSize());
        return platformData;
    });

    return FontRanges(FontCache::forCurrentThread().fontForPlatformData(originalPlatformData));
}

}
