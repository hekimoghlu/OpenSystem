/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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
#include "FontCascadeDescription.h"

#include "SystemFontDatabaseCoreText.h"

namespace WebCore {

static inline Vector<RetainPtr<CTFontDescriptorRef>> systemFontCascadeList(const FontDescription& description, const AtomString& cssFamily, SystemFontKind systemFontKind, AllowUserInstalledFonts allowUserInstalledFonts)
{
    return SystemFontDatabaseCoreText::forCurrentThread().cascadeList(description, cssFamily, systemFontKind, allowUserInstalledFonts);
}

unsigned FontCascadeDescription::effectiveFamilyCount() const
{
    // FIXME: Move all the other system font keywords from fontDescriptorWithFamilySpecialCase() to here.
    unsigned result = 0;
    for (unsigned i = 0; i < familyCount(); ++i) {
        const auto& cssFamily = familyAt(i);
        if (auto use = SystemFontDatabaseCoreText::forCurrentThread().matchSystemFontUse(cssFamily))
            result += systemFontCascadeList(*this, cssFamily, *use, shouldAllowUserInstalledFonts()).size();
        else
            ++result;
    }
    return result;
}

FontFamilySpecification FontCascadeDescription::effectiveFamilyAt(unsigned index) const
{
    // The special cases in this function need to match the behavior in FontCacheCoreText.cpp. This code
    // is used for regular (element style) lookups, and the code in FontDescriptionCocoa.cpp is used when
    // src:local(special-cased-name) is specified inside an @font-face block.
    // FIXME: Currently, an @font-face block corresponds to a single item in the font-family: fallback list, which
    // means that "src:local(system-ui)" can't follow the Core Text cascade list (the way it does for regular lookups).
    // These two behaviors should be unified, which would hopefully allow us to delete this duplicate code.
    for (unsigned i = 0; i < familyCount(); ++i) {
        const auto& cssFamily = familyAt(i);
        if (auto use = SystemFontDatabaseCoreText::forCurrentThread().matchSystemFontUse(cssFamily)) {
            auto cascadeList = systemFontCascadeList(*this, cssFamily, *use, shouldAllowUserInstalledFonts());
            if (index < cascadeList.size())
                return FontFamilySpecification(cascadeList[index].get());
            index -= cascadeList.size();
        }
        else if (!index)
            return cssFamily;
        else
            --index;
    }
    ASSERT_NOT_REACHED();
    return nullAtom();
}

AtomString FontDescription::platformResolveGenericFamily(UScriptCode script, const AtomString& locale, const AtomString& familyName)
{
    ASSERT((locale.isNull() && script == USCRIPT_COMMON) || !locale.isNull());
    if (script == USCRIPT_COMMON)
        return nullAtom();

    // FIXME: Use the system font database to handle standardFamily
    if (familyName == serifFamily)
        return AtomString { SystemFontDatabaseCoreText::forCurrentThread().serifFamily(locale.string()) };
    if (familyName == sansSerifFamily)
        return AtomString { SystemFontDatabaseCoreText::forCurrentThread().sansSerifFamily(locale.string()) };
    if (familyName == cursiveFamily)
        return AtomString { SystemFontDatabaseCoreText::forCurrentThread().cursiveFamily(locale.string()) };
    if (familyName == fantasyFamily)
        return AtomString { SystemFontDatabaseCoreText::forCurrentThread().fantasyFamily(locale.string()) };
    if (familyName == monospaceFamily)
        return AtomString { SystemFontDatabaseCoreText::forCurrentThread().monospaceFamily(locale.string()) };

    return nullAtom();
}

}

