/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#import "FontAttributeChanges.h"

#if PLATFORM(COCOA)

#import "Font.h"
#import "FontCache.h"
#import "FontCacheCoreText.h"
#import "FontDescription.h"

namespace WebCore {

const String& FontChanges::platformFontFamilyNameForCSS() const
{
    // The family name may not be specific enough to get us the font specified.
    // In some cases, the only way to get exactly what we are looking for is to use
    // the Postscript name. If we don't find a font with the same Postscript name,
    // then we'll have to use the Postscript name to make the CSS specific enough.
    // This logic was originally from WebHTMLView, and is now in WebCore so that it can
    // be shared between WebKitLegacy and WebKit.
    auto cfFontName = m_fontName.createCFString();
    if (fontNameIsSystemFont(cfFontName.get()))
        return m_fontFamily;
    RetainPtr<CFStringRef> fontNameFromDescription;

    FontDescription description;
    description.setIsItalic(m_italic.value_or(false));
    description.setWeight(FontSelectionValue { m_bold.value_or(false) ? 900 : 500 });
    if (auto font = FontCache::forCurrentThread().fontForFamily(description, m_fontFamily))
        fontNameFromDescription = adoptCF(CTFontCopyPostScriptName(font->getCTFont()));

    if (fontNameFromDescription && CFStringCompare(cfFontName.get(), fontNameFromDescription.get(), 0) == kCFCompareEqualTo)
        return m_fontFamily;

    return m_fontName;
}

} // namespace WebCore

#endif
