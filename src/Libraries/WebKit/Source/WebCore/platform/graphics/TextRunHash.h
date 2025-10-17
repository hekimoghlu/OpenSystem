/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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

#include "TextRun.h"
#include <wtf/HashFunctions.h>
#include <wtf/HashTraits.h>
#include <wtf/Hasher.h>

namespace WebCore {

inline void add(Hasher& hasher, const ExpansionBehavior& expansionBehavior)
{
    add(hasher, expansionBehavior.left, expansionBehavior.right);
}

inline void add(Hasher& hasher, const TextRun& textRun)
{
    add(hasher, textRun.m_text, textRun.m_tabSize, textRun.m_xpos, textRun.m_horizontalGlyphStretch, textRun.m_expansion, textRun.m_expansionBehavior, textRun.m_allowTabs, textRun.m_direction, textRun.m_directionalOverride, textRun.m_characterScanForCodePath, textRun.m_disableSpacing);
}

inline bool TextRun::operator==(const TextRun& other) const
{
    return m_text == other.m_text
        && m_tabSize == other.m_tabSize
        && m_xpos == other.m_xpos
        && m_horizontalGlyphStretch == other.m_horizontalGlyphStretch
        && m_expansion == other.m_expansion
        && m_expansionBehavior == other.m_expansionBehavior
        && m_allowTabs == other.m_allowTabs
        && m_direction == other.m_direction
        && m_directionalOverride == other.m_directionalOverride
        && m_characterScanForCodePath == other.m_characterScanForCodePath
        && m_disableSpacing == other.m_disableSpacing
        && m_textSpacingState == other.m_textSpacingState;
}

struct TextRunHash {
    static unsigned hash(const TextRun& textRun) { return computeHash(textRun); }
    static bool equal(const TextRun& a, const TextRun& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

} // namespace WebCore

namespace WTF {

template<> struct DefaultHash<WebCore::TextRun> : WebCore::TextRunHash { };

template<> struct HashTraits<WebCore::TextRun> : GenericHashTraits<WebCore::TextRun> {
    static bool isDeletedValue(const WebCore::TextRun& value) { return value.isHashTableDeletedValue(); }
    static bool isEmptyValue(const WebCore::TextRun& value) { return value.isHashTableEmptyValue(); }
    static void constructDeletedValue(WebCore::TextRun& slot) { new (NotNull, &slot) WebCore::TextRun(WTF::HashTableDeletedValue); }
    static WebCore::TextRun emptyValue() { return WebCore::TextRun(WTF::HashTableEmptyValue); }
};

} // namespace WTF
