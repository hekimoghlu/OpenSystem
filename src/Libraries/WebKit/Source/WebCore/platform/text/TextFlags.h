/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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

#include "FontTaggedSettings.h"
#include <optional>
#include <variant>
#include <vector>
#include <wtf/Hasher.h>
#include <wtf/Markable.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class FontFeatureValues;

enum class TextRenderingMode : uint8_t {
    AutoTextRendering,
    OptimizeSpeed,
    OptimizeLegibility,
    GeometricPrecision
};

enum class FontSmoothingMode : uint8_t {
    AutoSmoothing,
    NoSmoothing,
    Antialiased,
    SubpixelAntialiased
};

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, FontSmoothingMode);

enum class FontOrientation : bool {
    Horizontal,
    Vertical
};

enum class NonCJKGlyphOrientation : bool {
    Mixed,
    Upright
};

struct ExpansionBehavior {
    enum class Behavior : uint8_t {
        Forbid,
        Allow,
        Force
    };

    ExpansionBehavior()
        : left(Behavior::Forbid)
        , right(Behavior::Allow)
    {

    }

    ExpansionBehavior(Behavior left, Behavior right)
        : left(left)
        , right(right)
    {
    }

    friend bool operator==(const ExpansionBehavior&, const ExpansionBehavior&) = default;

    static ExpansionBehavior defaultBehavior()
    {
        return { };
    }

    static ExpansionBehavior allowRightOnly()
    {
        return { Behavior::Forbid, Behavior::Allow };
    }

    static ExpansionBehavior allowLeftOnly()
    {
        return { Behavior::Allow, Behavior::Forbid };
    }

    static ExpansionBehavior forceLeftOnly()
    {
        return { Behavior::Force, Behavior::Forbid };
    }

    static ExpansionBehavior forbidAll()
    {
        return { Behavior::Forbid, Behavior::Forbid };
    }

    static constexpr unsigned bitsOfKind = 2;
    Behavior left : bitsOfKind;
    Behavior right : bitsOfKind;
};

WTF::TextStream& operator<<(WTF::TextStream&, ExpansionBehavior::Behavior);
WTF::TextStream& operator<<(WTF::TextStream&, ExpansionBehavior);

enum class FontSynthesisLonghandValue : bool {
    None,
    Auto
};

enum class FontVariantLigatures : uint8_t { Normal, Yes, No };
enum class FontVariantPosition : uint8_t { Normal, Subscript, Superscript };

WTF::TextStream& operator<<(WTF::TextStream&, FontVariantPosition);

enum class FontVariantCaps : uint8_t {
    Normal,
    Small,
    AllSmall,
    Petite,
    AllPetite,
    Unicase,
    Titling
};

WTF::TextStream& operator<<(WTF::TextStream&, FontVariantCaps);

enum class FontVariantNumericFigure : uint8_t {
    Normal,
    LiningNumbers,
    OldStyleNumbers
};

enum class FontVariantNumericSpacing : uint8_t {
    Normal,
    ProportionalNumbers,
    TabularNumbers
};

enum class FontVariantNumericFraction : uint8_t {
    Normal,
    DiagonalFractions,
    StackedFractions
};

enum class FontVariantNumericOrdinal : bool { Normal, Yes };
enum class FontVariantNumericSlashedZero : bool { Normal, Yes };

struct FontVariantAlternatesValues {
    friend bool operator==(const FontVariantAlternatesValues&, const FontVariantAlternatesValues&) = default;

    String stylistic;
    Vector<String> styleset;
    Vector<String> characterVariant;
    String swash;
    String ornaments;
    String annotation;
    bool historicalForms = false;

    friend void add(Hasher&, const FontVariantAlternatesValues&);

    struct MarkableTraits {
        static bool isEmptyValue(const FontVariantAlternatesValues& value)
        {
            return value.m_isEmpty;
        }

        static FontVariantAlternatesValues emptyValue()
        {
            FontVariantAlternatesValues emptyValue;
            emptyValue.m_isEmpty = true;
            return emptyValue;
        }
    };

private:
    friend MarkableTraits;
    bool m_isEmpty { false };
};

class FontVariantAlternates {
    using Values = FontVariantAlternatesValues;

public:
    friend bool operator==(const FontVariantAlternates&, const FontVariantAlternates&) = default;

    bool isNormal() const
    {
        return !m_values;
    }

    const Values& values() const
    {
        ASSERT(!isNormal());
        return *m_values;
    }

    Values& valuesRef()
    {
        if (isNormal())
            setValues();

        return *m_values;
    }

    void setValues()
    {
        m_values = Values { };
    }

    static FontVariantAlternates Normal()
    {
        return { };
    }

    friend void add(Hasher&, const FontVariantAlternates&);

private:
    Markable<Values> m_values;
    FontVariantAlternates() = default;
};

WTF::TextStream& operator<<(WTF::TextStream&, const FontVariantAlternates&);

enum class FontVariantEastAsianVariant : uint8_t {
    Normal,
    Jis78,
    Jis83,
    Jis90,
    Jis04,
    Simplified,
    Traditional
};

enum class FontVariantEastAsianWidth : uint8_t {
    Normal,
    Full,
    Proportional
};

enum class FontVariantEastAsianRuby : uint8_t {
    Normal,
    Yes
};

enum class FontVariantEmoji : uint8_t {
    Normal,
    Text,
    Emoji,
    Unicode,
};

struct FontVariantSettings {
    FontVariantSettings()
        : commonLigatures(FontVariantLigatures::Normal)
        , discretionaryLigatures(FontVariantLigatures::Normal)
        , historicalLigatures(FontVariantLigatures::Normal)
        , contextualAlternates(FontVariantLigatures::Normal)
        , position(FontVariantPosition::Normal)
        , caps(FontVariantCaps::Normal)
        , numericFigure(FontVariantNumericFigure::Normal)
        , numericSpacing(FontVariantNumericSpacing::Normal)
        , numericFraction(FontVariantNumericFraction::Normal)
        , numericOrdinal(FontVariantNumericOrdinal::Normal)
        , numericSlashedZero(FontVariantNumericSlashedZero::Normal)
        , alternates(FontVariantAlternates::Normal())
        , eastAsianVariant(FontVariantEastAsianVariant::Normal)
        , eastAsianWidth(FontVariantEastAsianWidth::Normal)
        , eastAsianRuby(FontVariantEastAsianRuby::Normal)
        , emoji(FontVariantEmoji::Normal)
    {
    }

    FontVariantSettings(
        FontVariantLigatures commonLigatures,
        FontVariantLigatures discretionaryLigatures,
        FontVariantLigatures historicalLigatures,
        FontVariantLigatures contextualAlternates,
        FontVariantPosition position,
        FontVariantCaps caps,
        FontVariantNumericFigure numericFigure,
        FontVariantNumericSpacing numericSpacing,
        FontVariantNumericFraction numericFraction,
        FontVariantNumericOrdinal numericOrdinal,
        FontVariantNumericSlashedZero numericSlashedZero,
        FontVariantAlternates alternates,
        FontVariantEastAsianVariant eastAsianVariant,
        FontVariantEastAsianWidth eastAsianWidth,
        FontVariantEastAsianRuby eastAsianRuby,
        FontVariantEmoji emoji)
            : commonLigatures(commonLigatures)
            , discretionaryLigatures(discretionaryLigatures)
            , historicalLigatures(historicalLigatures)
            , contextualAlternates(contextualAlternates)
            , position(position)
            , caps(caps)
            , numericFigure(numericFigure)
            , numericSpacing(numericSpacing)
            , numericFraction(numericFraction)
            , numericOrdinal(numericOrdinal)
            , numericSlashedZero(numericSlashedZero)
            , alternates(alternates)
            , eastAsianVariant(eastAsianVariant)
            , eastAsianWidth(eastAsianWidth)
            , eastAsianRuby(eastAsianRuby)
            , emoji(emoji)
    {
    }

    bool isAllNormal() const
    {
        return commonLigatures == FontVariantLigatures::Normal
            && discretionaryLigatures == FontVariantLigatures::Normal
            && historicalLigatures == FontVariantLigatures::Normal
            && contextualAlternates == FontVariantLigatures::Normal
            && position == FontVariantPosition::Normal
            && caps == FontVariantCaps::Normal
            && numericFigure == FontVariantNumericFigure::Normal
            && numericSpacing == FontVariantNumericSpacing::Normal
            && numericFraction == FontVariantNumericFraction::Normal
            && numericOrdinal == FontVariantNumericOrdinal::Normal
            && numericSlashedZero == FontVariantNumericSlashedZero::Normal
            && alternates.isNormal()
            && eastAsianVariant == FontVariantEastAsianVariant::Normal
            && eastAsianWidth == FontVariantEastAsianWidth::Normal
            && eastAsianRuby == FontVariantEastAsianRuby::Normal
            && emoji == FontVariantEmoji::Normal;
    }

    friend bool operator==(const FontVariantSettings&, const FontVariantSettings&) = default;

    FontVariantLigatures commonLigatures;
    FontVariantLigatures discretionaryLigatures;
    FontVariantLigatures historicalLigatures;
    FontVariantLigatures contextualAlternates;
    FontVariantPosition position;
    FontVariantCaps caps;
    FontVariantNumericFigure numericFigure;
    FontVariantNumericSpacing numericSpacing;
    FontVariantNumericFraction numericFraction;
    FontVariantNumericOrdinal numericOrdinal;
    FontVariantNumericSlashedZero numericSlashedZero;
    FontVariantAlternates alternates;
    FontVariantEastAsianVariant eastAsianVariant;
    FontVariantEastAsianWidth eastAsianWidth;
    FontVariantEastAsianRuby eastAsianRuby;
    FontVariantEmoji emoji;
};

struct FontVariantLigaturesValues {
    FontVariantLigaturesValues(
        FontVariantLigatures commonLigatures,
        FontVariantLigatures discretionaryLigatures,
        FontVariantLigatures historicalLigatures,
        FontVariantLigatures contextualAlternates)
            : commonLigatures(commonLigatures)
            , discretionaryLigatures(discretionaryLigatures)
            , historicalLigatures(historicalLigatures)
            , contextualAlternates(contextualAlternates)
    {
    }

    FontVariantLigatures commonLigatures;
    FontVariantLigatures discretionaryLigatures;
    FontVariantLigatures historicalLigatures;
    FontVariantLigatures contextualAlternates;
};

struct FontVariantNumericValues {
    FontVariantNumericValues(
        FontVariantNumericFigure figure,
        FontVariantNumericSpacing spacing,
        FontVariantNumericFraction fraction,
        FontVariantNumericOrdinal ordinal,
        FontVariantNumericSlashedZero slashedZero)
            : figure(figure)
            , spacing(spacing)
            , fraction(fraction)
            , ordinal(ordinal)
            , slashedZero(slashedZero)
    {
    }

    FontVariantNumericFigure figure;
    FontVariantNumericSpacing spacing;
    FontVariantNumericFraction fraction;
    FontVariantNumericOrdinal ordinal;
    FontVariantNumericSlashedZero slashedZero;
};

struct FontVariantEastAsianValues {
    FontVariantEastAsianValues(
        FontVariantEastAsianVariant variant,
        FontVariantEastAsianWidth width,
        FontVariantEastAsianRuby ruby)
            : variant(variant)
            , width(width)
            , ruby(ruby)
    {
    }

    FontVariantEastAsianVariant variant;
    FontVariantEastAsianWidth width;
    FontVariantEastAsianRuby ruby;
};


enum class FontWidthVariant : uint8_t {
    RegularWidth,
    HalfWidth,
    ThirdWidth,
    QuarterWidth,
    LastFontWidthVariant = QuarterWidth
};

const unsigned FontWidthVariantWidth = 2;

static_assert(!(static_cast<unsigned>(FontWidthVariant::LastFontWidthVariant) >> FontWidthVariantWidth), "FontWidthVariantWidth is correct");

enum class FontSmallCaps : bool {
    Off,
    On
};

enum class Kerning : uint8_t {
    Auto,
    Normal,
    NoShift
};

WTF::TextStream& operator<<(WTF::TextStream&, Kerning);

enum class FontOpticalSizing : bool {
    Enabled,
    Disabled
};

WTF::TextStream& operator<<(WTF::TextStream&, FontOpticalSizing);

// https://www.microsoft.com/typography/otspec/fvar.htm#VAT
enum class FontStyleAxis : uint8_t {
    slnt,
    ital
};

enum class AllowUserInstalledFonts : bool { No, Yes };

using FeaturesMap = UncheckedKeyHashMap<FontTag, int, FourCharacterTagHash, FourCharacterTagHashTraits>;
FeaturesMap computeFeatureSettingsFromVariants(const FontVariantSettings&, RefPtr<FontFeatureValues>);

enum class ResolvedEmojiPolicy : uint8_t {
    NoPreference,
    RequireText,
    RequireEmoji,
};

enum class ColorGlyphType : uint8_t {
    Outline,
    Color,
};

} // namespace WebCore
