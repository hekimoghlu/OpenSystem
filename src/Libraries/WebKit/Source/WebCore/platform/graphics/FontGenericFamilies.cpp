/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
#include "FontGenericFamilies.h"

#include <wtf/CrossThreadCopier.h>
#include <wtf/Language.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FontGenericFamilies);

using namespace WebKitFontFamilyNames;

static bool setGenericFontFamilyForScript(ScriptFontFamilyMap& fontMap, const String& family, UScriptCode script)
{
    if (family.isEmpty())
        return fontMap.remove(static_cast<int>(script));
    auto& familyInMap = fontMap.add(static_cast<int>(script), String { }).iterator->value;
    if (familyInMap == family)
        return false;
    familyInMap = family;
    return true;
}

static const String& genericFontFamilyForScript(const ScriptFontFamilyMap& fontMap, UScriptCode script)
{
    ScriptFontFamilyMap::const_iterator it = fontMap.find(static_cast<int>(script));
    if (it != fontMap.end())
        return it->value;
    // Content using USCRIPT_HAN doesn't tell us if we should be using Simplified Chinese or Traditional Chinese. In the
    // absence of all other signals, we consult with the user's system preferences.
    if (script == USCRIPT_HAN) {
        it = fontMap.find(static_cast<int>(userPrefersSimplifiedChinese() ? USCRIPT_SIMPLIFIED_HAN : USCRIPT_TRADITIONAL_HAN));
        if (it != fontMap.end())
            return it->value;
    }
    if (script != USCRIPT_COMMON)
        return genericFontFamilyForScript(fontMap, USCRIPT_COMMON);
    return emptyAtom();
}

FontGenericFamilies::FontGenericFamilies() = default;

FontGenericFamilies FontGenericFamilies::isolatedCopy() const &
{
    FontGenericFamilies copy;
    copy.m_standardFontFamilyMap = crossThreadCopy(m_standardFontFamilyMap);
    copy.m_serifFontFamilyMap = crossThreadCopy(m_serifFontFamilyMap);
    copy.m_fixedFontFamilyMap = crossThreadCopy(m_fixedFontFamilyMap);
    copy.m_sansSerifFontFamilyMap = crossThreadCopy(m_sansSerifFontFamilyMap);
    copy.m_cursiveFontFamilyMap = crossThreadCopy(m_cursiveFontFamilyMap);
    copy.m_fantasyFontFamilyMap = crossThreadCopy(m_fantasyFontFamilyMap);
    copy.m_pictographFontFamilyMap = crossThreadCopy(m_pictographFontFamilyMap);
    return copy;
}

FontGenericFamilies FontGenericFamilies::isolatedCopy() &&
{
    FontGenericFamilies copy;
    copy.m_standardFontFamilyMap = crossThreadCopy(WTFMove(m_standardFontFamilyMap));
    copy.m_serifFontFamilyMap = crossThreadCopy(WTFMove(m_serifFontFamilyMap));
    copy.m_fixedFontFamilyMap = crossThreadCopy(WTFMove(m_fixedFontFamilyMap));
    copy.m_sansSerifFontFamilyMap = crossThreadCopy(WTFMove(m_sansSerifFontFamilyMap));
    copy.m_cursiveFontFamilyMap = crossThreadCopy(WTFMove(m_cursiveFontFamilyMap));
    copy.m_fantasyFontFamilyMap = crossThreadCopy(WTFMove(m_fantasyFontFamilyMap));
    copy.m_pictographFontFamilyMap = crossThreadCopy(WTFMove(m_pictographFontFamilyMap));
    return copy;
}

const String& FontGenericFamilies::standardFontFamily(UScriptCode script) const
{
    return genericFontFamilyForScript(m_standardFontFamilyMap, script);
}

const String& FontGenericFamilies::fixedFontFamily(UScriptCode script) const
{
    return genericFontFamilyForScript(m_fixedFontFamilyMap, script);
}

const String& FontGenericFamilies::serifFontFamily(UScriptCode script) const
{
    return genericFontFamilyForScript(m_serifFontFamilyMap, script);
}

const String& FontGenericFamilies::sansSerifFontFamily(UScriptCode script) const
{
    return genericFontFamilyForScript(m_sansSerifFontFamilyMap, script);
}

const String& FontGenericFamilies::cursiveFontFamily(UScriptCode script) const
{
    return genericFontFamilyForScript(m_cursiveFontFamilyMap, script);
}

const String& FontGenericFamilies::fantasyFontFamily(UScriptCode script) const
{
    return genericFontFamilyForScript(m_fantasyFontFamilyMap, script);
}

const String& FontGenericFamilies::pictographFontFamily(UScriptCode script) const
{
    return genericFontFamilyForScript(m_pictographFontFamilyMap, script);
}

bool FontGenericFamilies::setStandardFontFamily(const String& family, UScriptCode script)
{
    return setGenericFontFamilyForScript(m_standardFontFamilyMap, family, script);
}

bool FontGenericFamilies::setFixedFontFamily(const String& family, UScriptCode script)
{
    return setGenericFontFamilyForScript(m_fixedFontFamilyMap, family, script);
}

bool FontGenericFamilies::setSerifFontFamily(const String& family, UScriptCode script)
{
    return setGenericFontFamilyForScript(m_serifFontFamilyMap, family, script);
}

bool FontGenericFamilies::setSansSerifFontFamily(const String& family, UScriptCode script)
{
    return setGenericFontFamilyForScript(m_sansSerifFontFamilyMap, family, script);
}

bool FontGenericFamilies::setCursiveFontFamily(const String& family, UScriptCode script)
{
    return setGenericFontFamilyForScript(m_cursiveFontFamilyMap, family, script);
}

bool FontGenericFamilies::setFantasyFontFamily(const String& family, UScriptCode script)
{
    return setGenericFontFamilyForScript(m_fantasyFontFamilyMap, family, script);
}

bool FontGenericFamilies::setPictographFontFamily(const String& family, UScriptCode script)
{
    return setGenericFontFamilyForScript(m_pictographFontFamilyMap, family, script);
}

const String* FontGenericFamilies::fontFamily(FamilyNamesIndex family, UScriptCode script) const
{
    switch (family) {
    case FamilyNamesIndex::CursiveFamily:
        return &cursiveFontFamily(script);
    case FamilyNamesIndex::FantasyFamily:
        return &fantasyFontFamily(script);
    case FamilyNamesIndex::MonospaceFamily:
        return &fixedFontFamily(script);
    case FamilyNamesIndex::PictographFamily:
        return &pictographFontFamily(script);
    case FamilyNamesIndex::SansSerifFamily:
        return &sansSerifFontFamily(script);
    case FamilyNamesIndex::SerifFamily:
        return &serifFontFamily(script);
    case FamilyNamesIndex::StandardFamily:
        return &standardFontFamily(script);
    case FamilyNamesIndex::SystemUiFamily:
        return nullptr;
    }

    ASSERT_NOT_REACHED();
    return nullptr;
}

}
