/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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

#include <unicode/uchar.h>
#include <unicode/uscript.h>
#include <wtf/text/StringCommon.h>

namespace WTF {

inline bool isEmojiGroupCandidate(char32_t character)
{
#define SYMBOLS_AND_PICTOGRAPHS_EXTENDED_A 298
#if U_ICU_VERSION_MAJOR_NUM < 64
#define UBLOCK_SYMBOLS_AND_PICTOGRAPHS_EXTENDED_A SYMBOLS_AND_PICTOGRAPHS_EXTENDED_A
#else
static_assert(UBLOCK_SYMBOLS_AND_PICTOGRAPHS_EXTENDED_A == SYMBOLS_AND_PICTOGRAPHS_EXTENDED_A);
#endif

    switch (static_cast<int>(ublock_getCode(character))) {
    case UBLOCK_MISCELLANEOUS_SYMBOLS:
    case UBLOCK_DINGBATS:
    case UBLOCK_MISCELLANEOUS_SYMBOLS_AND_PICTOGRAPHS:
    case UBLOCK_EMOTICONS:
    case UBLOCK_TRANSPORT_AND_MAP_SYMBOLS:
    case UBLOCK_SUPPLEMENTAL_SYMBOLS_AND_PICTOGRAPHS:
    case UBLOCK_SYMBOLS_AND_PICTOGRAPHS_EXTENDED_A:
        return true;
    default:
        return false;
    }
}

inline bool isEmojiFitzpatrickModifier(char32_t character)
{
    // U+1F3FB - EMOJI MODIFIER FITZPATRICK TYPE-1-2
    // U+1F3FC - EMOJI MODIFIER FITZPATRICK TYPE-3
    // U+1F3FD - EMOJI MODIFIER FITZPATRICK TYPE-4
    // U+1F3FE - EMOJI MODIFIER FITZPATRICK TYPE-5
    // U+1F3FF - EMOJI MODIFIER FITZPATRICK TYPE-6

    return character >= 0x1F3FB && character <= 0x1F3FF;
}

inline bool isVariationSelector(char32_t character)
{
    return character >= 0xFE00 && character <= 0xFE0F;
}

inline bool isEmojiKeycapBase(char32_t character)
{
    return (character >= '0' && character <= '9') || character == '#' || character == '*';
}

inline bool isEmojiRegionalIndicator(char32_t character)
{
    return character >= 0x1F1E6 && character <= 0x1F1FF;
}

inline bool isEmojiWithPresentationByDefault(char32_t character)
{
    // No characters in Latin-1 include "Emoji_Presentation"
    // https://www.unicode.org/Public/UCD/latest/ucd/emoji/emoji-data.txt
    if (isLatin1(character))
        return false;
    return u_hasBinaryProperty(character, UCHAR_EMOJI_PRESENTATION);
}

inline bool isEmojiModifierBase(char32_t character)
{
    // No characters in Latin-1 include "Emoji_Modifier_Base"
    // https://www.unicode.org/Public/UCD/latest/ucd/emoji/emoji-data.txt
    if (isLatin1(character))
        return false;
    return u_hasBinaryProperty(character, UCHAR_EMOJI_MODIFIER_BASE);
}

inline bool isDefaultIgnorableCodePoint(char32_t character)
{
    return u_hasBinaryProperty(character, UCHAR_DEFAULT_IGNORABLE_CODE_POINT);
}

inline bool isControlCharacter(char32_t character)
{
    return u_charType(character) == U_CONTROL_CHAR;
}

inline bool isPrivateUseAreaCharacter(char32_t character)
{
    auto block = ublock_getCode(character);
    return block == UBLOCK_PRIVATE_USE_AREA || block == UBLOCK_SUPPLEMENTARY_PRIVATE_USE_AREA_A || block == UBLOCK_SUPPLEMENTARY_PRIVATE_USE_AREA_B;
}

inline bool isPunctuation(char32_t character)
{
    return U_GET_GC_MASK(character) & U_GC_P_MASK;
}

inline bool isOpeningPunctuation(uint32_t generalCategoryMask)
{
    return generalCategoryMask & U_GC_PS_MASK;
}

inline bool isClosingPunctuation(uint32_t generalCategoryMask)
{
    return generalCategoryMask & U_GC_PE_MASK;
}

inline bool isOfScriptType(char32_t codePoint, UScriptCode scriptType)
{
    UErrorCode error = U_ZERO_ERROR;
    UScriptCode script = uscript_getScript(codePoint, &error);
    if (error != U_ZERO_ERROR) {
        LOG_ERROR("got ICU error while trying to look at scripts: %d", error);
        return false;
    }
    return script == scriptType;
}

inline UEastAsianWidth eastAsianWidth(char32_t character)
{
    return static_cast<UEastAsianWidth>(u_getIntPropertyValue(character, UCHAR_EAST_ASIAN_WIDTH));
}

inline bool isEastAsianFullWidth(char32_t character)
{
    return eastAsianWidth(character) == UEastAsianWidth::U_EA_FULLWIDTH;
}

inline bool isCJKSymbolOrPunctuation(char32_t character)
{
    // CJK Symbols and Punctuation block (U+3000â€“U+303F)
    return character >= 0x3000 && character <= 0x303F;
}

inline bool isFullwidthMiddleDotPunctuation(char32_t character)
{
    // U+00B7 MIDDLE DOT
    // U+2027 HYPHENATION POINT
    // U+30FB KATAKANA MIDDLE DOT
    return character == 0x00B7 || character == 0x2027 || character == 0x30FB;
}

inline bool isCombiningMark(char32_t character)
{
    return 0x0300 <= character && character <= 0x036F;
}

} // namespace WTF

using WTF::isEmojiGroupCandidate;
using WTF::isEmojiFitzpatrickModifier;
using WTF::isVariationSelector;
using WTF::isEmojiKeycapBase;
using WTF::isEmojiRegionalIndicator;
using WTF::isEmojiWithPresentationByDefault;
using WTF::isEmojiModifierBase;
using WTF::isDefaultIgnorableCodePoint;
using WTF::isControlCharacter;
using WTF::isPrivateUseAreaCharacter;
using WTF::isPunctuation;
using WTF::isOpeningPunctuation;
using WTF::isClosingPunctuation;
using WTF::isOfScriptType;
using WTF::isEastAsianFullWidth;
using WTF::isCJKSymbolOrPunctuation;
using WTF::isFullwidthMiddleDotPunctuation;
using WTF::isCombiningMark;
