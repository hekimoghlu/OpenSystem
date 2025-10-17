/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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

#include "FontPalette.h"
#include "FontSelectionAlgorithm.h"
#include "FontSizeAdjust.h"
#include "FontTaggedSettings.h"
#include "TextFlags.h"
#include "TextSpacing.h"
#include "WebKitFontFamilyNames.h"
#include <unicode/uscript.h>
#include <wtf/MathExtras.h>

namespace WebCore {

using namespace WebKitFontFamilyNames;

class FontDescription {
public:
    WEBCORE_EXPORT FontDescription();

    friend bool operator==(const FontDescription&, const FontDescription&) = default;

    float computedSize() const { return m_computedSize; }
    // Adjusted size regarding @font-face size-adjust but not regarding font-size-adjust. The latter adjustment is done with updateSizeWithFontSizeAdjust() after the font's creation.
    float adjustedSizeForFontFace(float) const;
    std::optional<FontSelectionValue> italic() const { return m_fontSelectionRequest.slope; }
    FontSelectionValue width() const { return m_fontSelectionRequest.width; }
    FontSelectionValue weight() const { return m_fontSelectionRequest.weight; }
    const FontSelectionRequest& fontSelectionRequest() const { return m_fontSelectionRequest; }
    TextRenderingMode textRenderingMode() const { return static_cast<TextRenderingMode>(m_textRendering); }
    TextSpacingTrim textSpacingTrim() const { return m_textSpacingTrim; }
    TextAutospace textAutospace() const { return m_textAutospace; }
    UScriptCode script() const { return static_cast<UScriptCode>(m_script); }
    const AtomString& computedLocale() const { return m_locale; } // This is what you should be using for things like text shaping and font fallback
    const AtomString& specifiedLocale() const { return m_specifiedLocale; } // This is what you should be using for web-exposed things like -webkit-locale

    FontOrientation orientation() const { return static_cast<FontOrientation>(m_orientation); }
    NonCJKGlyphOrientation nonCJKGlyphOrientation() const { return static_cast<NonCJKGlyphOrientation>(m_nonCJKGlyphOrientation); }
    FontWidthVariant widthVariant() const { return static_cast<FontWidthVariant>(m_widthVariant); }
    const FontFeatureSettings& featureSettings() const { return m_featureSettings; }
    const FontVariationSettings& variationSettings() const { return m_variationSettings; }
    FontSynthesisLonghandValue fontSynthesisWeight() const { return static_cast<FontSynthesisLonghandValue>(m_fontSynthesisWeight); }
    FontSynthesisLonghandValue fontSynthesisStyle() const { return static_cast<FontSynthesisLonghandValue>(m_fontSynthesisStyle); }
    FontSynthesisLonghandValue fontSynthesisSmallCaps() const { return static_cast<FontSynthesisLonghandValue>(m_fontSynthesisCaps); }
    bool hasAutoFontSynthesisWeight() const { return fontSynthesisWeight() == FontSynthesisLonghandValue::Auto; }
    bool hasAutoFontSynthesisStyle() const { return fontSynthesisStyle() == FontSynthesisLonghandValue::Auto; }
    bool hasAutoFontSynthesisSmallCaps() const { return fontSynthesisSmallCaps() == FontSynthesisLonghandValue::Auto; }
    FontVariantLigatures variantCommonLigatures() const { return static_cast<FontVariantLigatures>(m_variantCommonLigatures); }
    FontVariantLigatures variantDiscretionaryLigatures() const { return static_cast<FontVariantLigatures>(m_variantDiscretionaryLigatures); }
    FontVariantLigatures variantHistoricalLigatures() const { return static_cast<FontVariantLigatures>(m_variantHistoricalLigatures); }
    FontVariantLigatures variantContextualAlternates() const { return static_cast<FontVariantLigatures>(m_variantContextualAlternates); }
    FontVariantPosition variantPosition() const { return static_cast<FontVariantPosition>(m_variantPosition); }
    FontVariantCaps variantCaps() const { return static_cast<FontVariantCaps>(m_variantCaps); }
    FontVariantNumericFigure variantNumericFigure() const { return static_cast<FontVariantNumericFigure>(m_variantNumericFigure); }
    FontVariantNumericSpacing variantNumericSpacing() const { return static_cast<FontVariantNumericSpacing>(m_variantNumericSpacing); }
    FontVariantNumericFraction variantNumericFraction() const { return static_cast<FontVariantNumericFraction>(m_variantNumericFraction); }
    FontVariantNumericOrdinal variantNumericOrdinal() const { return static_cast<FontVariantNumericOrdinal>(m_variantNumericOrdinal); }
    FontVariantNumericSlashedZero variantNumericSlashedZero() const { return static_cast<FontVariantNumericSlashedZero>(m_variantNumericSlashedZero); }
    const FontVariantAlternates& variantAlternates() const { return m_variantAlternates; }
    FontVariantEastAsianVariant variantEastAsianVariant() const { return static_cast<FontVariantEastAsianVariant>(m_variantEastAsianVariant); }
    FontVariantEastAsianWidth variantEastAsianWidth() const { return static_cast<FontVariantEastAsianWidth>(m_variantEastAsianWidth); }
    FontVariantEastAsianRuby variantEastAsianRuby() const { return static_cast<FontVariantEastAsianRuby>(m_variantEastAsianRuby); }
    FontVariantEmoji variantEmoji() const { return static_cast<FontVariantEmoji>(m_variantEmoji); }
    FontVariantSettings variantSettings() const
    {
        return { variantCommonLigatures(),
            variantDiscretionaryLigatures(),
            variantHistoricalLigatures(),
            variantContextualAlternates(),
            variantPosition(),
            variantCaps(),
            variantNumericFigure(),
            variantNumericSpacing(),
            variantNumericFraction(),
            variantNumericOrdinal(),
            variantNumericSlashedZero(),
            variantAlternates(),
            variantEastAsianVariant(),
            variantEastAsianWidth(),
            variantEastAsianRuby(),
            variantEmoji() };
    }
    FontOpticalSizing opticalSizing() const { return static_cast<FontOpticalSizing>(m_opticalSizing); }
    FontStyleAxis fontStyleAxis() const { return m_fontStyleAxis ? FontStyleAxis::ital : FontStyleAxis::slnt; }
    AllowUserInstalledFonts shouldAllowUserInstalledFonts() const { return static_cast<AllowUserInstalledFonts>(m_shouldAllowUserInstalledFonts); }
    bool shouldDisableLigaturesForSpacing() const { return m_shouldDisableLigaturesForSpacing; }
    const FontPalette& fontPalette() const { return m_fontPalette; }
    FontSizeAdjust fontSizeAdjust() const { return m_sizeAdjust; }

    void setComputedSize(float s) { m_computedSize = clampToFloat(s); }
    void setTextSpacingTrim(TextSpacingTrim v) { m_textSpacingTrim = v; }
    void setTextAutospace(TextAutospace v) { m_textAutospace = v; }
    void setItalic(std::optional<FontSelectionValue> italic) { m_fontSelectionRequest.slope = italic; }
    void setWidth(FontSelectionValue width) { m_fontSelectionRequest.width = width; }
    void setIsItalic(bool isItalic) { setItalic(isItalic ? std::optional<FontSelectionValue> { italicValue() } : std::optional<FontSelectionValue> { }); }
    void setWeight(FontSelectionValue weight) { m_fontSelectionRequest.weight = weight; }
    void setTextRenderingMode(TextRenderingMode rendering) { m_textRendering = enumToUnderlyingType(rendering); }
    void setOrientation(FontOrientation orientation) { m_orientation = enumToUnderlyingType(orientation); }
    void setNonCJKGlyphOrientation(NonCJKGlyphOrientation orientation) { m_nonCJKGlyphOrientation = enumToUnderlyingType(orientation); }
    void setWidthVariant(FontWidthVariant widthVariant) { m_widthVariant = enumToUnderlyingType(widthVariant); } // Make sure new callers of this sync with FontPlatformData::isForTextCombine()!
    void setSpecifiedLocale(const AtomString&);
    void setFeatureSettings(FontFeatureSettings&& settings) { m_featureSettings = WTFMove(settings); }
    void setVariationSettings(FontVariationSettings&& settings) { m_variationSettings = WTFMove(settings); }
    void setFontSynthesisWeight(FontSynthesisLonghandValue value) { m_fontSynthesisWeight =  enumToUnderlyingType(value); }
    void setFontSynthesisStyle(FontSynthesisLonghandValue value) { m_fontSynthesisStyle = enumToUnderlyingType(value); }
    void setFontSynthesisSmallCaps(FontSynthesisLonghandValue value) { m_fontSynthesisCaps = enumToUnderlyingType(value); }
    void setVariantCommonLigatures(FontVariantLigatures variant) { m_variantCommonLigatures = enumToUnderlyingType(variant); }
    void setVariantDiscretionaryLigatures(FontVariantLigatures variant) { m_variantDiscretionaryLigatures = enumToUnderlyingType(variant); }
    void setVariantHistoricalLigatures(FontVariantLigatures variant) { m_variantHistoricalLigatures = enumToUnderlyingType(variant); }
    void setVariantContextualAlternates(FontVariantLigatures variant) { m_variantContextualAlternates = enumToUnderlyingType(variant); }
    void setVariantPosition(FontVariantPosition variant) { m_variantPosition = enumToUnderlyingType(variant); }
    void setVariantCaps(FontVariantCaps variant) { m_variantCaps = enumToUnderlyingType(variant); }
    void setVariantNumericFigure(FontVariantNumericFigure variant) { m_variantNumericFigure = enumToUnderlyingType(variant); }
    void setVariantNumericSpacing(FontVariantNumericSpacing variant) { m_variantNumericSpacing = enumToUnderlyingType(variant); }
    void setVariantNumericFraction(FontVariantNumericFraction variant) { m_variantNumericFraction = enumToUnderlyingType(variant); }
    void setVariantNumericOrdinal(FontVariantNumericOrdinal variant) { m_variantNumericOrdinal = enumToUnderlyingType(variant); }
    void setVariantNumericSlashedZero(FontVariantNumericSlashedZero variant) { m_variantNumericSlashedZero = enumToUnderlyingType(variant); }
    void setVariantAlternates(const FontVariantAlternates& variant) { m_variantAlternates = variant; }
    void setVariantAlternates(FontVariantAlternates&& variant) { m_variantAlternates = WTFMove(variant); }
    void setVariantEastAsianVariant(FontVariantEastAsianVariant variant) { m_variantEastAsianVariant = enumToUnderlyingType(variant); }
    void setVariantEastAsianWidth(FontVariantEastAsianWidth variant) { m_variantEastAsianWidth = enumToUnderlyingType(variant); }
    void setVariantEastAsianRuby(FontVariantEastAsianRuby variant) { m_variantEastAsianRuby = enumToUnderlyingType(variant); }
    void setVariantEmoji(FontVariantEmoji variant) { m_variantEmoji = enumToUnderlyingType(variant); }
    void setOpticalSizing(FontOpticalSizing sizing) { m_opticalSizing = enumToUnderlyingType(sizing); }
    void setFontStyleAxis(FontStyleAxis axis) { m_fontStyleAxis = axis == FontStyleAxis::ital; }
    void setShouldAllowUserInstalledFonts(AllowUserInstalledFonts shouldAllowUserInstalledFonts) { m_shouldAllowUserInstalledFonts = enumToUnderlyingType(shouldAllowUserInstalledFonts); }
    void setShouldDisableLigaturesForSpacing(bool shouldDisableLigaturesForSpacing) { m_shouldDisableLigaturesForSpacing = shouldDisableLigaturesForSpacing; }
    void setFontPalette(const FontPalette& fontPalette) { m_fontPalette = fontPalette; }
    void setFontSizeAdjust(FontSizeAdjust fontSizeAdjust) { m_sizeAdjust = fontSizeAdjust; }

    static AtomString platformResolveGenericFamily(UScriptCode, const AtomString& locale, const AtomString& familyName);

private:
    // FIXME: Investigate moving these into their own object on the heap (to save memory).
    FontFeatureSettings m_featureSettings;
    FontVariationSettings m_variationSettings;
    FontVariantAlternates m_variantAlternates;
    FontPalette m_fontPalette;
    FontSizeAdjust m_sizeAdjust;
    AtomString m_locale;
    AtomString m_specifiedLocale;

    FontSelectionRequest m_fontSelectionRequest;
    TextSpacingTrim m_textSpacingTrim;
    TextAutospace m_textAutospace;
    float m_computedSize { 0 }; // Computed size adjusted for the minimum font size and the zoom factor.
    unsigned m_orientation : 1; // FontOrientation - Whether the font is rendering on a horizontal line or a vertical line.
    unsigned m_nonCJKGlyphOrientation : 1; // NonCJKGlyphOrientation - Only used by vertical text. Determines the default orientation for non-ideograph glyphs.
    unsigned m_widthVariant : 2; // FontWidthVariant
    unsigned m_textRendering : 2; // TextRenderingMode
    unsigned m_script : 7; // Used to help choose an appropriate font for generic font families.
    unsigned m_fontSynthesisWeight : 1;
    unsigned m_fontSynthesisStyle : 1;
    unsigned m_fontSynthesisCaps : 1;
    unsigned m_variantCommonLigatures : 2; // FontVariantLigatures
    unsigned m_variantDiscretionaryLigatures : 2; // FontVariantLigatures
    unsigned m_variantHistoricalLigatures : 2; // FontVariantLigatures
    unsigned m_variantContextualAlternates : 2; // FontVariantLigatures
    unsigned m_variantPosition : 2; // FontVariantPosition
    unsigned m_variantCaps : 3; // FontVariantCaps
    unsigned m_variantNumericFigure : 2; // FontVariantNumericFigure
    unsigned m_variantNumericSpacing : 2; // FontVariantNumericSpacing
    unsigned m_variantNumericFraction : 2; // FontVariantNumericFraction
    unsigned m_variantNumericOrdinal : 1; // FontVariantNumericOrdinal
    unsigned m_variantNumericSlashedZero : 1; // FontVariantNumericSlashedZero
    unsigned m_variantEastAsianVariant : 3; // FontVariantEastAsianVariant
    unsigned m_variantEastAsianWidth : 2; // FontVariantEastAsianWidth
    unsigned m_variantEastAsianRuby : 1; // FontVariantEastAsianRuby
    unsigned m_variantEmoji : 2; // FontVariantEmoji
    unsigned m_opticalSizing : 1; // FontOpticalSizing
    unsigned m_fontStyleAxis : 1; // Whether "font-style: italic" or "font-style: oblique 20deg" was specified
    unsigned m_shouldAllowUserInstalledFonts : 1; // AllowUserInstalledFonts: If this description is allowed to match a user-installed font
    unsigned m_shouldDisableLigaturesForSpacing : 1; // If letter-spacing is nonzero, we need to disable ligatures, which affects font preparation
};

}
