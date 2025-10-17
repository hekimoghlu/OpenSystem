/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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
#pragma once

#include "WebKitFontFamilyNames.h"
#include <unicode/uscript.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

// UScriptCode uses -1 and 0 for UScriptInvalidCode and UScriptCommon.
// We need to use -2 and -3 for empty value and deleted value.
struct UScriptCodeHashTraits : HashTraits<int> {
    static const bool emptyValueIsZero = false;
    static int emptyValue() { return -2; }
    static void constructDeletedValue(int& slot) { slot = -3; }
    static bool isDeletedValue(int value) { return value == -3; }
};

typedef UncheckedKeyHashMap<int, String, DefaultHash<int>, UScriptCodeHashTraits> ScriptFontFamilyMap;

class FontGenericFamilies {
    WTF_MAKE_TZONE_ALLOCATED(FontGenericFamilies);
public:
    FontGenericFamilies();

    FontGenericFamilies isolatedCopy() const &;
    FontGenericFamilies isolatedCopy() &&;

    const String& standardFontFamily(UScriptCode = USCRIPT_COMMON) const;
    const String& fixedFontFamily(UScriptCode = USCRIPT_COMMON) const;
    const String& serifFontFamily(UScriptCode = USCRIPT_COMMON) const;
    const String& sansSerifFontFamily(UScriptCode = USCRIPT_COMMON) const;
    const String& cursiveFontFamily(UScriptCode = USCRIPT_COMMON) const;
    const String& fantasyFontFamily(UScriptCode = USCRIPT_COMMON) const;
    const String& pictographFontFamily(UScriptCode = USCRIPT_COMMON) const;

    const String* fontFamily(WebKitFontFamilyNames::FamilyNamesIndex, UScriptCode = USCRIPT_COMMON) const;

    bool setStandardFontFamily(const String&, UScriptCode);
    bool setFixedFontFamily(const String&, UScriptCode);
    bool setSerifFontFamily(const String&, UScriptCode);
    bool setSansSerifFontFamily(const String&, UScriptCode);
    bool setCursiveFontFamily(const String&, UScriptCode);
    bool setFantasyFontFamily(const String&, UScriptCode);
    bool setPictographFontFamily(const String&, UScriptCode);

private:
    ScriptFontFamilyMap m_standardFontFamilyMap;
    ScriptFontFamilyMap m_serifFontFamilyMap;
    ScriptFontFamilyMap m_fixedFontFamilyMap;
    ScriptFontFamilyMap m_sansSerifFontFamilyMap;
    ScriptFontFamilyMap m_cursiveFontFamilyMap;
    ScriptFontFamilyMap m_fantasyFontFamilyMap;
    ScriptFontFamilyMap m_pictographFontFamilyMap;
};

}
