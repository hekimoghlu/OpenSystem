/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
#include "SystemFontDatabase.h"

#include "WebKitFontFamilyNames.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

SystemFontDatabase& SystemFontDatabase::singleton()
{
    static NeverDestroyed<SystemFontDatabase> database = SystemFontDatabase();
    return database.get();
}

static const float defaultControlFontPixelSize = 13;

auto SystemFontDatabase::platformSystemFontShorthandInfo(FontShorthand fontShorthand) -> SystemFontShorthandInfo
{
    static bool initialized;
    static NONCLIENTMETRICS ncm;

    if (!initialized) {
        initialized = true;
        ncm.cbSize = sizeof(NONCLIENTMETRICS);
        ::SystemParametersInfo(SPI_GETNONCLIENTMETRICS, sizeof(ncm), &ncm, 0);
    }

    LOGFONT logFont;
    bool shouldUseDefaultControlFontPixelSize = false;
    switch (fontShorthand) {
    case FontShorthand::Icon:
        ::SystemParametersInfo(SPI_GETICONTITLELOGFONT, sizeof(logFont), &logFont, 0);
        break;
    case FontShorthand::Menu:
        logFont = ncm.lfMenuFont;
        break;
    case FontShorthand::MessageBox:
        logFont = ncm.lfMessageFont;
        break;
    case FontShorthand::StatusBar:
        logFont = ncm.lfStatusFont;
        break;
    case FontShorthand::Caption:
        logFont = ncm.lfCaptionFont;
        break;
    case FontShorthand::SmallCaption:
        logFont = ncm.lfSmCaptionFont;
        break;
    case FontShorthand::WebkitSmallControl:
    case FontShorthand::WebkitMiniControl: // Just map to small.
    case FontShorthand::WebkitControl: // Just map to small.
        shouldUseDefaultControlFontPixelSize = true;
        FALLTHROUGH;
    default: { // Everything else uses the stock GUI font.
        HGDIOBJ hGDI = ::GetStockObject(DEFAULT_GUI_FONT);
        if (!hGDI)
            return { WebKitFontFamilyNames::standardFamily, 16, normalWeightValue() };
        if (::GetObject(hGDI, sizeof(logFont), &logFont) <= 0)
            return { WebKitFontFamilyNames::standardFamily, 16, normalWeightValue() };
    }
    }
    float size = shouldUseDefaultControlFontPixelSize ? defaultControlFontPixelSize : std::abs(logFont.lfHeight);
    auto weight = logFont.lfWeight >= 700 ? boldWeightValue() : normalWeightValue();
    return { logFont.lfFaceName, size, weight };
}

void SystemFontDatabase::platformInvalidate()
{
}

} // namespace WebCore
