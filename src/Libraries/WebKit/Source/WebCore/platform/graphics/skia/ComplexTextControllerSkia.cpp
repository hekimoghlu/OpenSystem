/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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
#include "ComplexTextController.h"

#include "FontCascade.h"
#include "FontFeatureValues.h"
#include "FontTaggedSettings.h"
#include "HbUniquePtr.h"
#include "SurrogatePairAwareTextIterator.h"
#include "text/TextFlags.h"
#include <hb-icu.h>
#include <hb-ot.h>

namespace WebCore {

static inline float harfBuzzPositionToFloat(hb_position_t value)
{
    return static_cast<float>(value) / (1 << 16);
}

ComplexTextController::ComplexTextRun::ComplexTextRun(hb_buffer_t* buffer, const Font& font, std::span<const UChar> characters, unsigned stringLocation, unsigned indexBegin, unsigned indexEnd)
    : m_initialAdvance(0, 0)
    , m_font(font)
    , m_characters(characters)
    , m_indexBegin(indexBegin)
    , m_indexEnd(indexEnd)
    , m_glyphCount(hb_buffer_get_length(buffer))
    , m_stringLocation(stringLocation)
    , m_isLTR(HB_DIRECTION_IS_FORWARD(hb_buffer_get_direction(buffer)))
    , m_textAutospaceSize(TextAutospace::textAutospaceSize(font))
{
    if (!m_glyphCount)
        return;

    m_glyphs.grow(m_glyphCount);
    m_baseAdvances.grow(m_glyphCount);
    m_glyphOrigins.grow(m_glyphCount);
    m_coreTextIndices.grow(m_glyphCount);

    WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GLib/Win port
    hb_glyph_info_t* glyphInfos = hb_buffer_get_glyph_infos(buffer, nullptr);
    hb_glyph_position_t* glyphPositions = hb_buffer_get_glyph_positions(buffer, nullptr);
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

    // HarfBuzz returns the shaping result in visual order. We don't need to flip for RTL.
    for (unsigned i = 0; i < m_glyphCount; ++i) {
        m_coreTextIndices[i] = glyphInfos[i].cluster;

        uint16_t glyph = glyphInfos[i].codepoint;
        if (m_font.isZeroWidthSpaceGlyph(glyph) || !m_font.platformData().size()) {
            m_glyphs[i] = glyph;
            m_baseAdvances[i] = { };
            m_glyphOrigins[i] = { };
            continue;
        }

        float offsetX = harfBuzzPositionToFloat(glyphPositions[i].x_offset);
        float offsetY = harfBuzzPositionToFloat(glyphPositions[i].y_offset);
        float advanceX = harfBuzzPositionToFloat(glyphPositions[i].x_advance);
        float advanceY = harfBuzzPositionToFloat(glyphPositions[i].y_advance);

        m_glyphs[i] = glyph;
        m_baseAdvances[i] = { advanceX, advanceY };
        m_glyphOrigins[i] = { offsetX, offsetY };
    }
    m_initialAdvance = toFloatSize(m_glyphOrigins[0]);
}

static std::optional<UScriptCode> characterScript(char32_t character)
{
    UErrorCode errorCode = U_ZERO_ERROR;
    UScriptCode script = uscript_getScript(character, &errorCode);
    if (U_FAILURE(errorCode))
        return std::nullopt;
    return script;
}

struct HBRun {
    unsigned startIndex;
    unsigned endIndex;
    UScriptCode script;
};

static std::optional<HBRun> findNextRun(std::span<const UChar> characters, unsigned offset)
{
    SurrogatePairAwareTextIterator textIterator(characters.subspan(offset), offset, characters.size());
    char32_t character;
    unsigned clusterLength = 0;
    if (!textIterator.consume(character, clusterLength))
        return std::nullopt;

    auto currentScript = characterScript(character);
    if (!currentScript)
        return std::nullopt;

    unsigned startIndex = offset;
    for (textIterator.advance(clusterLength); textIterator.consume(character, clusterLength); textIterator.advance(clusterLength)) {
        if (FontCascade::treatAsZeroWidthSpace(character))
            continue;

        auto nextScript = characterScript(character);
        if (!nextScript)
            return std::nullopt;

        // Â§5.1 Handling Characters with the Common Script Property.
        // Programs must resolve any of the special Script property values, such as Common,
        // based on the context of the surrounding characters. A simple heuristic uses the
        // script of the preceding character, which works well in many cases.
        // http://www.unicode.org/reports/tr24/#Common.
        //
        // FIXME: cover all other cases mentioned in the spec (ie. brackets or quotation marks).
        // https://bugs.webkit.org/show_bug.cgi?id=177003.
        //
        // If next script is inherited or common, keep using the current script.
        if (nextScript == USCRIPT_INHERITED || nextScript == USCRIPT_COMMON)
            continue;
        // If current script is inherited or common, set the next script as current.
        if (currentScript == USCRIPT_INHERITED || currentScript == USCRIPT_COMMON) {
            currentScript = nextScript;
            continue;
        }

        if (currentScript != nextScript && !uscript_hasScript(character, currentScript.value()))
            return std::optional<HBRun>({ startIndex, textIterator.currentIndex(), currentScript.value() });
    }

    return std::optional<HBRun>({ startIndex, textIterator.currentIndex(), currentScript.value() });
}

void ComplexTextController::collectComplexTextRunsForCharacters(std::span<const UChar> characters, unsigned stringLocation, const Font* font)
{
    if (!font) {
        // Create a run of missing glyphs from the primary font.
        m_complexTextRuns.append(ComplexTextRun::create(m_font.primaryFont(), characters, stringLocation, 0, characters.size(), m_run.ltr()));
        return;
    }

    Vector<HBRun> runList;
    size_t offset = 0;
    while (offset < characters.size()) {
        auto run = findNextRun(characters, offset);
        if (!run)
            break;
        runList.append(run.value());
        offset = run->endIndex;
    }

    size_t runCount = runList.size();
    if (!runCount)
        return;

    const auto& fontPlatformData = font->platformData();
    auto* hbFont = fontPlatformData.hbFont();
    RELEASE_ASSERT(hbFont);

    const auto& features = fontPlatformData.features();
    // Kerning is not handled as font features, so only in case it's explicitly disabled
    // we need to create a new vector to include kern feature.
    const hb_feature_t* featuresData = features.isEmpty() ? nullptr : features.data();
    unsigned featuresSize = features.size();
    Vector<hb_feature_t> featuresWithKerning;
    if (!m_font.enableKerning()) {
        featuresWithKerning.reserveInitialCapacity(featuresSize + 1);
        featuresWithKerning.append({ HB_TAG('k', 'e', 'r', 'n'), 0, 0, static_cast<unsigned>(-1) });
        featuresWithKerning.appendVector(features);
        featuresData = featuresWithKerning.data();
        featuresSize = featuresWithKerning.size();
    }

    HbUniquePtr<hb_buffer_t> buffer(hb_buffer_create());
    for (unsigned i = 0; i < runCount; ++i) {
        auto& run = runList[m_run.rtl() ? runCount - i - 1 : i];

        hb_buffer_set_script(buffer.get(), hb_icu_script_to_script(run.script));

        if (!m_mayUseNaturalWritingDirection || m_run.directionalOverride())
            hb_buffer_set_direction(buffer.get(), m_run.rtl() ? HB_DIRECTION_RTL : HB_DIRECTION_LTR);
        else
            hb_buffer_set_direction(buffer.get(), hb_script_get_horizontal_direction(hb_icu_script_to_script(run.script)));

        hb_buffer_add_utf16(buffer.get(), reinterpret_cast<const uint16_t*>(characters.data()), characters.size(), run.startIndex, run.endIndex - run.startIndex);

        hb_shape(hbFont, buffer.get(), featuresData, featuresSize);
        m_complexTextRuns.append(ComplexTextRun::create(buffer.get(), *font, characters, stringLocation, run.startIndex, run.endIndex));
        hb_buffer_reset(buffer.get());
    }
}

} // namespace WebCore
