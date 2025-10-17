/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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
#include "FontDescription.h"

#include "FontCascadeDescription.h"
#include "LocaleToScriptMapping.h"
#include <wtf/Language.h>

namespace WebCore {

FontDescription::FontDescription()
    : m_variantAlternates(FontCascadeDescription::initialVariantAlternates())
    , m_fontPalette({ FontPalette::Type::Normal, nullAtom() })
    , m_fontSelectionRequest { FontCascadeDescription::initialWeight(), FontCascadeDescription::initialWidth(), FontCascadeDescription::initialItalic() }
    , m_orientation(enumToUnderlyingType(FontOrientation::Horizontal))
    , m_nonCJKGlyphOrientation(enumToUnderlyingType(NonCJKGlyphOrientation::Mixed))
    , m_widthVariant(enumToUnderlyingType(FontWidthVariant::RegularWidth))
    , m_textRendering(enumToUnderlyingType(TextRenderingMode::AutoTextRendering))
    , m_script(USCRIPT_COMMON)
    , m_fontSynthesisWeight(enumToUnderlyingType(FontSynthesisLonghandValue::Auto))
    , m_fontSynthesisStyle(enumToUnderlyingType(FontSynthesisLonghandValue::Auto))
    , m_fontSynthesisCaps(enumToUnderlyingType(FontSynthesisLonghandValue::Auto))
    , m_variantCommonLigatures(enumToUnderlyingType(FontVariantLigatures::Normal))
    , m_variantDiscretionaryLigatures(enumToUnderlyingType(FontVariantLigatures::Normal))
    , m_variantHistoricalLigatures(enumToUnderlyingType(FontVariantLigatures::Normal))
    , m_variantContextualAlternates(enumToUnderlyingType(FontVariantLigatures::Normal))
    , m_variantPosition(enumToUnderlyingType(FontVariantPosition::Normal))
    , m_variantCaps(enumToUnderlyingType(FontVariantCaps::Normal))
    , m_variantNumericFigure(enumToUnderlyingType(FontVariantNumericFigure::Normal))
    , m_variantNumericSpacing(enumToUnderlyingType(FontVariantNumericSpacing::Normal))
    , m_variantNumericFraction(enumToUnderlyingType(FontVariantNumericFraction::Normal))
    , m_variantNumericOrdinal(enumToUnderlyingType(FontVariantNumericOrdinal::Normal))
    , m_variantNumericSlashedZero(enumToUnderlyingType(FontVariantNumericSlashedZero::Normal))
    , m_variantEastAsianVariant(enumToUnderlyingType(FontVariantEastAsianVariant::Normal))
    , m_variantEastAsianWidth(enumToUnderlyingType(FontVariantEastAsianWidth::Normal))
    , m_variantEastAsianRuby(enumToUnderlyingType(FontVariantEastAsianRuby::Normal))
    , m_variantEmoji(enumToUnderlyingType(FontVariantEmoji::Normal))
    , m_opticalSizing(enumToUnderlyingType(FontOpticalSizing::Enabled))
    , m_fontStyleAxis(FontCascadeDescription::initialFontStyleAxis() == FontStyleAxis::ital)
    , m_shouldAllowUserInstalledFonts(enumToUnderlyingType(AllowUserInstalledFonts::No))
    , m_shouldDisableLigaturesForSpacing(false)
{
}

static AtomString computeSpecializedChineseLocale()
{
    // FIXME: This is not passing ShouldMinimizeLanguages::No and then getting minimized languages,
    // which may cause the matching below to fail.
    for (auto& language : userPreferredLanguages()) {
        if (startsWithLettersIgnoringASCIICase(language, "zh-"_s))
            return AtomString { language };
    }
    return "zh-hans"_s; // We have no signal. Pick one option arbitrarily.
}

static AtomString& cachedSpecializedChineseLocale()
{
    static MainThreadNeverDestroyed<AtomString> specializedChineseLocale;
    return specializedChineseLocale.get();
}

static void fontDescriptionLanguageChanged(void*)
{
    cachedSpecializedChineseLocale() = computeSpecializedChineseLocale();
}

static const AtomString& specializedChineseLocale()
{
    auto& locale = cachedSpecializedChineseLocale();
    if (cachedSpecializedChineseLocale().isNull()) {
        static char forNonNullPointer;
        addLanguageChangeObserver(&forNonNullPointer, &fontDescriptionLanguageChanged); // We will never remove the observer, so all we need is a non-null pointer.
        fontDescriptionLanguageChanged(nullptr);
    }
    return locale;
}

void FontDescription::setSpecifiedLocale(const AtomString& locale)
{
    ASSERT(isMainThread());
    m_specifiedLocale = locale;
    m_script = localeToScriptCodeForFontSelection(m_specifiedLocale);
    m_locale = m_script == USCRIPT_HAN ? specializedChineseLocale() : m_specifiedLocale;
}

#if !PLATFORM(COCOA)
AtomString FontDescription::platformResolveGenericFamily(UScriptCode, const AtomString&, const AtomString&)
{
    return nullAtom();
}
#endif

float FontDescription::adjustedSizeForFontFace(float fontFaceSizeAdjust) const
{
    // It is not worth modifying the used size with @font-face size-adjust if we are to re-adjust it later with font-size-adjust. This is because font-size-adjust will overrule this change, since size-adjust also modifies the font's metric values and thus, keeps the aspect-value unchanged.
    return fontSizeAdjust().value ? computedSize() : fontFaceSizeAdjust * computedSize();

}
} // namespace WebCore
