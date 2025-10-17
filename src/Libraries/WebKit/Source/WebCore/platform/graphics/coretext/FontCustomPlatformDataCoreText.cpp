/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#include "FontCustomPlatformData.h"

#include "CSSFontFaceSrcValue.h"
#include "Font.h"
#include "FontCache.h"
#include "FontCacheCoreText.h"
#include "FontCreationContext.h"
#include "FontDescription.h"
#include "FontPlatformData.h"
#include "SharedBuffer.h"
#include "StyleFontSizeFunctions.h"
#include "UnrealizedCoreTextFont.h"
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreText/CoreText.h>
#include <pal/cf/CoreTextSoftLink.h>
#include <pal/spi/cf/CoreTextSPI.h>

namespace WebCore {

FontCustomPlatformData::~FontCustomPlatformData() = default;

FontPlatformData FontCustomPlatformData::fontPlatformData(const FontDescription& fontDescription, bool bold, bool italic, const FontCreationContext& fontCreationContext)
{
    auto size = fontDescription.adjustedSizeForFontFace(fontCreationContext.sizeAdjust());
    UnrealizedCoreTextFont unrealizedFont = { RetainPtr { fontDescriptor } };
    unrealizedFont.setSize(size);
    unrealizedFont.modify([&](CFMutableDictionaryRef attributes) {
        addAttributesForWebFonts(attributes, fontDescription.shouldAllowUserInstalledFonts());
    });

    FontOrientation orientation = fontDescription.orientation();
    FontWidthVariant widthVariant = fontDescription.widthVariant();

    auto font = preparePlatformFont(WTFMove(unrealizedFont), fontDescription, fontCreationContext);
    ASSERT(font);
    FontPlatformData platformData(font.get(), size, bold, italic, orientation, widthVariant, fontDescription.textRenderingMode(), this);

    platformData.updateSizeWithFontSizeAdjust(fontDescription.fontSizeAdjust(), fontDescription.computedSize());
    return platformData;
}

static RetainPtr<CFDataRef> extractFontCustomPlatformData(SharedBuffer& buffer, const String& itemInCollection)
{
    RetainPtr<CFDataRef> bufferData = buffer.createCFData();

    FPFontRef font = nullptr;
    auto array = adoptCF(FPFontCreateFontsFromData(bufferData.get()));
    if (!array)
        return nullptr;
    auto length = CFArrayGetCount(array.get());
    if (length <= 0)
        return nullptr;
    if (!itemInCollection.isNull()) {
        if (auto desiredName = itemInCollection.createCFString()) {
            for (CFIndex i = 0; i < length; ++i) {
                auto candidate = static_cast<FPFontRef>(CFArrayGetValueAtIndex(array.get(), i));
                auto postScriptName = adoptCF(FPFontCopyPostScriptName(candidate));
                if (CFStringCompare(postScriptName.get(), desiredName.get(), 0) == kCFCompareEqualTo) {
                    font = candidate;
                    break;
                }
            }
        }
    }
    if (!font)
        font = static_cast<FPFontRef>(CFArrayGetValueAtIndex(array.get(), 0));

    // Retain the extracted font contents, so the GPU process doesn't have to extract it a second time later.
    // This is a power optimization.
    return adoptCF(FPFontCopySFNTData(font));
}

RefPtr<FontCustomPlatformData> FontCustomPlatformData::create(SharedBuffer& buffer, const String& itemInCollection)
{
    RetainPtr extractedData = extractFontCustomPlatformData(buffer, itemInCollection);
    if (!extractedData) {
        // Something is wrong with the font.
        return nullptr;
    }

    RetainPtr fontDescriptor = adoptCF(CTFontManagerCreateFontDescriptorFromData(extractedData.get()));
    Ref bufferRef = SharedBuffer::create(extractedData.get());

    FontPlatformData::CreationData creationData = { WTFMove(bufferRef), itemInCollection };
    return adoptRef(new FontCustomPlatformData(fontDescriptor.get(), WTFMove(creationData)));
}

RefPtr<FontCustomPlatformData> FontCustomPlatformData::createMemorySafe(SharedBuffer& buffer, const String& itemInCollection)
{
    if (!PAL::canLoad_CoreText_CTFontManagerCreateMemorySafeFontDescriptorFromData())
        return nullptr;

    RetainPtr extractedData = extractFontCustomPlatformData(buffer, itemInCollection);
    if (!extractedData) {
        // Something is wrong with the font.
        return nullptr;
    }

    RetainPtr fontDescriptor = adoptCF(PAL::softLinkCoreTextCTFontManagerCreateMemorySafeFontDescriptorFromData(extractedData.get()));

    // Safe Font parser could not handle this font. This is already logged by CachedFontLoadRequest::ensureCustomFontData
    if (!fontDescriptor)
        return nullptr;

    Ref bufferRef = SharedBuffer::create(extractedData.get());

    FontPlatformData::CreationData creationData = { WTFMove(bufferRef), itemInCollection };
    return adoptRef(new FontCustomPlatformData(fontDescriptor.get(), WTFMove(creationData)));
}

std::optional<Ref<FontCustomPlatformData>> FontCustomPlatformData::tryMakeFromSerializationData(FontCustomPlatformSerializedData&& data, bool shouldUseLockdownFontParser )
{
    RefPtr fontCustomPlatformData = shouldUseLockdownFontParser ? FontCustomPlatformData::createMemorySafe(WTFMove(data.fontFaceData), data.itemInCollection) : FontCustomPlatformData::create(WTFMove(data.fontFaceData), data.itemInCollection);
    if (!fontCustomPlatformData)
        return std::nullopt;
    fontCustomPlatformData->m_renderingResourceIdentifier = data.renderingResourceIdentifier;
    return fontCustomPlatformData.releaseNonNull();
}

FontCustomPlatformSerializedData FontCustomPlatformData::serializedData() const
{
    return FontCustomPlatformSerializedData { creationData.fontFaceData, creationData.itemInCollection, m_renderingResourceIdentifier };
}

bool FontCustomPlatformData::supportsFormat(const String& format)
{
    return equalLettersIgnoringASCIICase(format, "truetype"_s)
        || equalLettersIgnoringASCIICase(format, "opentype"_s)
        || equalLettersIgnoringASCIICase(format, "woff2"_s)
        || equalLettersIgnoringASCIICase(format, "woff2-variations"_s)
        || equalLettersIgnoringASCIICase(format, "woff-variations"_s)
        || equalLettersIgnoringASCIICase(format, "truetype-variations"_s)
        || equalLettersIgnoringASCIICase(format, "opentype-variations"_s)
        || equalLettersIgnoringASCIICase(format, "woff"_s)
        || equalLettersIgnoringASCIICase(format, "svg"_s);
}

bool FontCustomPlatformData::supportsTechnology(const FontTechnology& tech)
{
    switch (tech) {
    case FontTechnology::ColorColrv0:
    case FontTechnology::ColorSbix:
    case FontTechnology::ColorSvg:
    case FontTechnology::FeaturesAat:
    case FontTechnology::FeaturesOpentype:
    case FontTechnology::Palettes:
    case FontTechnology::Variations:
        return true;
    default:
        return false;
    }
}

}
