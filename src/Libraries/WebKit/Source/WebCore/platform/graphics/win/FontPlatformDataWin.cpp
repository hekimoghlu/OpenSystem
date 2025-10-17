/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 15, 2024.
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

#include "FontCustomPlatformData.h"
#include "HWndDC.h"
#include "SharedBuffer.h"
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

FontPlatformData::FontPlatformData(GDIObject<HFONT> font, float size, bool bold, bool oblique, const FontCustomPlatformData* customPlatformData)
    : FontPlatformData(size, bold, oblique, FontOrientation::Horizontal, FontWidthVariant::RegularWidth, TextRenderingMode::AutoTextRendering, customPlatformData)
{
    m_hfont = SharedGDIObject<HFONT>::create(WTFMove(font));
    platformDataInit(m_hfont->get(), size);
}

RefPtr<SharedBuffer> FontPlatformData::platformOpenTypeTable(uint32_t table) const
{
    HWndDC hdc(0);
    HGDIOBJ oldFont = SelectObject(hdc, hfont());

    DWORD size = GetFontData(hdc, table, 0, 0, 0);
    RefPtr<SharedBuffer> buffer;
    if (size != GDI_ERROR) {
        Vector<uint8_t> data(size);
        DWORD result = GetFontData(hdc, table, 0, (PVOID)data.data(), size);
        ASSERT_UNUSED(result, result == size);
        buffer = SharedBuffer::create(WTFMove(data));
    }

    SelectObject(hdc, oldFont);
    return buffer;
}

FontPlatformData FontPlatformData::create(const Attributes& data, const FontCustomPlatformData* custom)
{
    LOGFONT logFont = data.m_font;
    if (custom)
        wcscpy_s(logFont.lfFaceName, LF_FACESIZE, custom->name.wideCharacters().data());

    auto gdiFont = adoptGDIObject(CreateFontIndirect(&logFont));
    return FontPlatformData(WTFMove(gdiFont), data.m_size, data.m_syntheticBold, data.m_syntheticOblique, custom);
}

FontPlatformData::Attributes FontPlatformData::attributes() const
{
    Attributes result(m_size, m_orientation, m_widthVariant, m_textRenderingMode, m_syntheticBold, m_syntheticOblique);

    GetObject(hfont(), sizeof(LOGFONT), &result.m_font);
    return result;
}

std::optional<FontPlatformData> FontPlatformData::fromIPCData(float size, FontOrientation&& orientation, FontWidthVariant&& widthVariant, TextRenderingMode&& textRenderingMode, bool syntheticBold, bool syntheticOblique, IPCData&& ipcData)
{
    return WTF::switchOn(ipcData,
        [&] (const FontPlatformSerializedData& d) -> std::optional<FontPlatformData> {
            if (auto gdiFont = adoptGDIObject(CreateFontIndirect(&d.logFont)))
                return FontPlatformData(WTFMove(gdiFont), size, syntheticBold, syntheticOblique);

            return std::nullopt;
        },
        [&] (FontPlatformSerializedCreationData& d) -> std::optional<FontPlatformData> {
            auto fontFaceData = SharedBuffer::create(WTFMove(d.fontFaceData));
            if (RefPtr fontCustomPlatformData = FontCustomPlatformData::create(fontFaceData, d.itemInCollection))
                return FontPlatformData(size, syntheticBold, syntheticOblique, WTFMove(orientation), WTFMove(widthVariant), WTFMove(textRenderingMode), fontCustomPlatformData.get());

            return std::nullopt;
        }
    );
}

FontPlatformData::IPCData FontPlatformData::toIPCData() const
{
    if (auto* data = creationData())
        return FontPlatformSerializedCreationData { { data->fontFaceData->span() }, data->itemInCollection };

    LOGFONT logFont;
    GetObject(hfont(), sizeof logFont, &logFont);
    return FontPlatformSerializedData { WTFMove(logFont) };
}

}
