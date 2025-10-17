/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#ifndef WidthIterator_h
#define WidthIterator_h

#include "GlyphBuffer.h"
#include "WritingMode.h"
#include <unicode/umachine.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class FontCascade;
class FontCascadeDescription;
class Font;
class TextRun;
struct GlyphData;
struct GlyphIndexRange;
struct OriginalAdvancesForCharacterTreatedAsSpace;
struct AdvanceInternalState;
struct SmallCapsState;

using CharactersTreatedAsSpace = Vector<OriginalAdvancesForCharacterTreatedAsSpace, 64>;

struct WidthIterator {
    WTF_MAKE_TZONE_ALLOCATED(WidthIterator);
public:
    WidthIterator(const FontCascade&, const TextRun&, SingleThreadWeakHashSet<const Font>* fallbackFonts = 0, bool accountForGlyphBounds = false, bool forTextEmphasis = false);

    void advance(unsigned to, GlyphBuffer&);
    bool advanceOneCharacter(float& width, GlyphBuffer&);
    void finalize(GlyphBuffer&);

    float maxGlyphBoundingBoxY() const { ASSERT(m_accountForGlyphBounds); return m_maxGlyphBoundingBoxY; }
    float minGlyphBoundingBoxY() const { ASSERT(m_accountForGlyphBounds); return m_minGlyphBoundingBoxY; }
    float firstGlyphOverflow() const { ASSERT(m_accountForGlyphBounds); return m_firstGlyphOverflow; }
    float lastGlyphOverflow() const { ASSERT(m_accountForGlyphBounds); return m_lastGlyphOverflow; }

    const TextRun& run() const { return m_run; }
    float runWidthSoFar() const { return m_runWidthSoFar; }
    unsigned currentCharacterIndex() const { return m_currentCharacterIndex; }

    WEBCORE_EXPORT static bool characterCanUseSimplifiedTextMeasuring(char32_t, bool whitespaceIsCollapsed);

private:
    GlyphData glyphDataForCharacter(char32_t, bool mirror);
    template <typename TextIterator>
    inline void advanceInternal(TextIterator&, GlyphBuffer&);

    enum class TransformsType { None, Forced, NotForced };
    TransformsType shouldApplyFontTransforms(const GlyphBuffer&, unsigned lastGlyphCount, unsigned currentCharacterIndex) const;
    struct ApplyFontTransformsResult {
        float additionalAdvance;
        GlyphBufferAdvance initialAdvance;
    };
    ApplyFontTransformsResult applyFontTransforms(GlyphBuffer&, unsigned lastGlyphCount, const Font&, CharactersTreatedAsSpace&);
    void commitCurrentFontRange(AdvanceInternalState&);
    void startNewFontRangeIfNeeded(AdvanceInternalState&, SmallCapsState&, const FontCascadeDescription&);
    void applyInitialAdvance(GlyphBuffer&, GlyphBufferAdvance initialAdvance, unsigned lastGlyphCount);

    bool hasExtraSpacing() const;
    void applyExtraSpacingAfterShaping(GlyphBuffer&, unsigned characterStartIndex, unsigned glyphBufferStartIndex, unsigned characterDestinationIndex, float startingRunWidth);
    void applyCSSVisibilityRules(GlyphBuffer&, unsigned glyphBufferStartIndex);

    struct AdditionalWidth {
        float left;
        float right;
        float leftExpansion;
        float rightExpansion;
    };
    AdditionalWidth calculateAdditionalWidth(GlyphBuffer&, GlyphBufferStringOffset currentCharacterIndex, unsigned leadingGlyphIndex, unsigned trailingGlyphIndex, float position) const;
    void applyAdditionalWidth(GlyphBuffer&, GlyphIndexRange, float leftAdditionalWidth, float rightAdditionalWidth, float leftExpansionAdditionalWidth, float rightExpansionAdditionalWidth);

    TextDirection direction() const { return m_direction; }
    bool rtl() const { return m_direction == TextDirection::RTL; }
    bool ltr() const { return m_direction == TextDirection::LTR; }

    CheckedRef<const FontCascade> m_font;
    CheckedRef<const TextRun> m_run;
    SingleThreadWeakHashSet<const Font>* m_fallbackFonts { nullptr };

    std::optional<unsigned> m_lastCharacterIndex;
    GlyphBufferAdvance m_leftoverInitialAdvance { makeGlyphBufferAdvance() };
    unsigned m_currentCharacterIndex { 0 };
    float m_leftoverJustificationWidth { 0 };
    float m_runWidthSoFar { 0 };
    float m_expansion { 0 };
    float m_expansionPerOpportunity { 0 };
    float m_maxGlyphBoundingBoxY { std::numeric_limits<float>::lowest() };
    float m_minGlyphBoundingBoxY { std::numeric_limits<float>::max() };
    float m_firstGlyphOverflow { 0 };
    float m_lastGlyphOverflow { 0 };
    TextDirection m_direction { TextDirection::LTR };
    bool m_containsTabs { false };
    bool m_isAfterExpansion { false };
    bool m_accountForGlyphBounds { false };
    bool m_enableKerning { false };
    bool m_requiresShaping { false };
    bool m_forTextEmphasis { false };
};

} // namespace WebCore

#endif
