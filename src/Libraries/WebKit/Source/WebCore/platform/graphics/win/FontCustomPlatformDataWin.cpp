/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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

#if PLATFORM(WIN)

#include "CSSFontFaceSrcValue.h"
#include "FontCreationContext.h"
#include "FontDescription.h"
#include "FontMemoryResource.h"
#include "FontPlatformData.h"
#include "OpenTypeUtilities.h"
#include "SharedBuffer.h"
#include <wtf/RetainPtr.h>
#include <wtf/text/Base64.h>
#include <wtf/win/GDIObject.h>

namespace WebCore {

FontCustomPlatformData::FontCustomPlatformData(const String& name, FontPlatformData::CreationData&& creationData)
    : name(name)
    , creationData(WTFMove(creationData))
    , m_renderingResourceIdentifier(RenderingResourceIdentifier::generate())
{
}

FontCustomPlatformData::~FontCustomPlatformData() = default;

static String createUniqueFontName()
{
    GUID fontUuid;
    CoCreateGuid(&fontUuid);

    auto fontName = base64EncodeToString({ reinterpret_cast<const uint8_t*>(&fontUuid), sizeof(fontUuid) });
    ASSERT(fontName.length() < LF_FACESIZE);
    return fontName;
}

RefPtr<FontCustomPlatformData> FontCustomPlatformData::create(SharedBuffer& buffer, const String& itemInCollection)
{
    String fontName = createUniqueFontName();
    auto fontResource = renameAndActivateFont(buffer, fontName);

    if (!fontResource)
        return nullptr;

    FontPlatformData::CreationData creationData = { buffer, itemInCollection, fontResource.releaseNonNull() };
    return adoptRef(new FontCustomPlatformData(fontName, WTFMove(creationData)));
}

RefPtr<FontCustomPlatformData> FontCustomPlatformData::createMemorySafe(SharedBuffer&, const String&)
{
    return nullptr;
}

bool FontCustomPlatformData::supportsFormat(const String& format)
{
    return equalLettersIgnoringASCIICase(format, "truetype"_s)
        || equalLettersIgnoringASCIICase(format, "opentype"_s)
#if USE(WOFF2)
        || equalLettersIgnoringASCIICase(format, "woff2"_s)
#endif
        || equalLettersIgnoringASCIICase(format, "woff"_s)
        || equalLettersIgnoringASCIICase(format, "svg"_s);
}

bool FontCustomPlatformData::supportsTechnology(const FontTechnology& tech)
{
    switch (tech) {
    case FontTechnology::ColorCbdt:
    case FontTechnology::ColorColrv0:
    case FontTechnology::ColorSbix:
    case FontTechnology::ColorSvg:
    case FontTechnology::FeaturesOpentype:
        return true;
    default:
        return false;
    }
}

std::optional<Ref<FontCustomPlatformData>> FontCustomPlatformData::tryMakeFromSerializationData(FontCustomPlatformSerializedData&& data, bool)
{
    RefPtr fontCustomPlatformData = FontCustomPlatformData::create(WTFMove(data.fontFaceData), data.itemInCollection);
    if (!fontCustomPlatformData)
        return std::nullopt;
    fontCustomPlatformData->m_renderingResourceIdentifier = data.renderingResourceIdentifier;
    return fontCustomPlatformData.releaseNonNull();
}

FontCustomPlatformSerializedData FontCustomPlatformData::serializedData() const
{
    return FontCustomPlatformSerializedData { creationData.fontFaceData, creationData.itemInCollection, m_renderingResourceIdentifier };
}

} // namespace WebCore

#endif // PLATFORM(WIN)
