/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 2, 2023.
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
#include <unicode/umachine.h>
#include <wtf/Forward.h>
#include <wtf/text/CharacterProperties.h>
#include <wtf/text/TextStream.h>
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

class Font;
class TextSpacingTrim;

namespace TextSpacing {

enum class CharacterClass : uint8_t {
    Undefined = 0,
    Ideograph = 1 << 0,
    NonIdeographLetter = 1 << 1,
    NonIdeographNumeral = 1 << 2,
    FullWidthOpeningPunctuation = 1 << 3,
    FullWidthClosingPunctuation = 1 << 4,
    FullWidthMiddleDotPunctuation = 1 << 5,
    FullWidthColonPunctuation = 1 << 6,
    FullWidthDotPunctuation = 1 << 7
};

struct CharactersData {
    char32_t previousCharacter { 0 };
    char32_t currentCharacter { 0 };
    char32_t nextCharacter { 0 };
    CharacterClass previousCharacterClass { };
    CharacterClass currentCharacterClass { };
    CharacterClass nextCharacterClass { };
};

// Classes are defined at https://www.w3.org/TR/css-text-4/#text-spacing-classes
CharacterClass characterClass(char32_t character);
struct SpacingState {
    bool operator==(const SpacingState&) const = default;
    CharacterClass lastCharacterClassFromPreviousRun { CharacterClass::Undefined };
};

bool isIdeograph(char32_t character);

RefPtr<Font> getHalfWidthFontIfNeeded(const Font&, const TextSpacingTrim&, CharactersData&);
} // namespace TextSpacing

class TextSpacingTrim {
public:
    enum class TrimType : uint8_t {
        SpaceAll = 0, // equivalent to None in text-spacing shorthand
        TrimAll,
        Auto
    };

    TextSpacingTrim() = default;
    TextSpacingTrim(TrimType trimType)
        : m_trim(trimType)
        { }

    bool isAuto() const { return m_trim == TrimType::Auto; }
    bool isSpaceAll() const { return m_trim == TrimType::SpaceAll; }
    bool shouldTrimSpacing(const TextSpacing::CharactersData&) const;
    friend bool operator==(const TextSpacingTrim&, const TextSpacingTrim&) = default;
    TrimType type() const { return m_trim; }
private:
    TrimType m_trim { TrimType::SpaceAll };
};

inline WTF::TextStream& operator<<(WTF::TextStream& ts, const TextSpacingTrim& value)
{
    // FIXME: add remaining values;
    switch (value.type()) {
    case TextSpacingTrim::TrimType::Auto:
        return ts << "auto";
    case TextSpacingTrim::TrimType::SpaceAll:
        return ts << "space-all";
    case TextSpacingTrim::TrimType::TrimAll:
        return ts << "trim-all";
    }
    return ts;
}

class TextAutospace {
public:
    enum class Type: uint8_t {
        Auto = 1 << 0,
        IdeographAlpha = 1 << 1,
        IdeographNumeric = 1 << 2,
        Normal = 1 << 3
    };

    using Options = OptionSet<Type>;

    TextAutospace() = default;
    TextAutospace(Options options)
        : m_options(options)
        { }

    bool isAuto() const { return m_options.contains(Type::Auto); }
    bool isNoAutospace() const { return m_options.isEmpty(); }
    bool isNormal() const { return m_options.contains(Type::Normal); }
    bool hasIdeographAlpha() const { return m_options.containsAny({ Type::IdeographAlpha, Type::Normal }); }
    bool hasIdeographNumeric() const { return m_options.containsAny({ Type::IdeographNumeric, Type::Normal }); }
    Options options() { return m_options; }
    friend bool operator==(const TextAutospace&, const TextAutospace&) = default;
    bool shouldApplySpacing(TextSpacing::CharacterClass firstCharacterClass, TextSpacing::CharacterClass secondCharacterClass) const;
    bool shouldApplySpacing(char32_t firstCharacter, char32_t secondCharacter) const;
    static float textAutospaceSize(const Font&);

private:
    Options m_options { };
};

inline WTF::TextStream& operator<<(WTF::TextStream& ts, const TextAutospace& value)
{
    // FIXME: add remaining values;
    if (value.isAuto())
        return ts << "auto";
    if (value.isNoAutospace())
        return ts << "no-autospace";
    if (value.isNormal())
        return ts << "normal";
    if (value.hasIdeographAlpha())
        ts << "ideograph-alpha";
    if (value.hasIdeographNumeric())
        ts << "ideograph-numeric";
    return ts;
}


} // namespace WebCore
