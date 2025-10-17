/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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

#include "PlatformScreen.h"
#include "SystemSettings.h"
#include "WebKitFontFamilyNames.h"
#include <gtk/gtk.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebCore {

SystemFontDatabase& SystemFontDatabase::singleton()
{
    static NeverDestroyed<SystemFontDatabase> database = SystemFontDatabase();
    return database.get();
}

auto SystemFontDatabase::platformSystemFontShorthandInfo(FontShorthand) -> SystemFontShorthandInfo
{
    // This will be a font selection string like "Sans 10" so we cannot use it as the family name.
    auto fontName = SystemSettings::singleton().fontName();
    if (!fontName || fontName->isEmpty())
        return { WebKitFontFamilyNames::standardFamily, 16, normalWeightValue() };

    PangoFontDescription* pangoDescription = pango_font_description_from_string(fontName->utf8().data());
    if (!pangoDescription)
        return { WebKitFontFamilyNames::standardFamily, 16, normalWeightValue() };

    int size = pango_font_description_get_size(pangoDescription) / PANGO_SCALE;
    // If the size of the font is in points, we need to convert it to pixels.
    if (!pango_font_description_get_size_is_absolute(pangoDescription))
        size = size * (fontDPI() / 72.0);

    SystemFontShorthandInfo result { AtomString::fromLatin1(pango_font_description_get_family(pangoDescription)), static_cast<float>(size), normalWeightValue() };
    pango_font_description_free(pangoDescription);
    return result;
}

void SystemFontDatabase::platformInvalidate()
{
}

} // namespace WebCore
