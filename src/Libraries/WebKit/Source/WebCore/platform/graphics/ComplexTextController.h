/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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

#include "FloatPoint.h"
#include "GlyphBuffer.h"
#include "TextSpacing.h"
#include <wtf/HashSet.h>
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

typedef unsigned short CGGlyph;

typedef const struct __CTRun * CTRunRef;
typedef const struct __CTLine * CTLineRef;

typedef struct hb_buffer_t hb_buffer_t;

namespace WTF {
class CachedTextBreakIterator;
}

namespace WebCore {

class FontCascade;
class Font;
class TextRun;

enum class GlyphIterationStyle : bool { IncludePartialGlyphs, ByWholeGlyphs };

// See https://trac.webkit.org/wiki/ComplexTextController for more information about ComplexTextController.
class ComplexTextController {
    WTF_MAKE_TZONE_ALLOCATED(ComplexTextController);
public:
    ComplexTextController(const FontCascade&, const TextRun&, bool mayUseNaturalWritingDirection = false, SingleThreadWeakHashSet<const Font>* fallbackFonts = 0, bool forTextEmphasis = false);

    static std::pair<float, float> enclosingGlyphBoundsForTextRun(const FontCascade&, const TextRun&);

    class ComplexTextRun;
    WEBCORE_EXPORT ComplexTextController(const FontCascade&, const TextRun&, Vector<Ref<ComplexTextRun>>&);

    // Advance and emit glyphs up to the specified character.
    WEBCORE_EXPORT void advance(unsigned to, GlyphBuffer* = nullptr, GlyphIterationStyle = GlyphIterationStyle::IncludePartialGlyphs, SingleThreadWeakHashSet<const Font>* fallbackFonts = nullptr);

    // Compute the character offset for a given x coordinate.
    unsigned offsetForPosition(float x, bool includePartialGlyphs);

    // Returns the width of everything we've consumed so far.
    float runWidthSoFar() const { return m_runWidthSoFar; }

    FloatSize totalAdvance() const { return m_totalAdvance; }

    float minGlyphBoundingBoxX() const { return m_minGlyphBoundingBoxX; }
    float maxGlyphBoundingBoxX() const { return m_maxGlyphBoundingBoxX; }
    float minGlyphBoundingBoxY() const { return m_minGlyphBoundingBoxY; }
    float maxGlyphBoundingBoxY() const { return m_maxGlyphBoundingBoxY; }

    class ComplexTextRun : public RefCounted<ComplexTextRun> {
    public:
        static Ref<ComplexTextRun> create(CTRunRef ctRun, const Font& font, std::span<const UChar> characters, unsigned stringLocation, unsigned indexBegin, unsigned indexEnd)
        {
            return adoptRef(*new ComplexTextRun(ctRun, font, characters, stringLocation, indexBegin, indexEnd));
        }

        static Ref<ComplexTextRun> create(hb_buffer_t* buffer, const Font& font, std::span<const UChar> characters, unsigned stringLocation, unsigned indexBegin, unsigned indexEnd)
        {
            return adoptRef(*new ComplexTextRun(buffer, font, characters, stringLocation, indexBegin, indexEnd));
        }

        static Ref<ComplexTextRun> create(const Font& font, std::span<const UChar> characters, unsigned stringLocation, unsigned indexBegin, unsigned indexEnd, bool ltr)
        {
            return adoptRef(*new ComplexTextRun(font, characters, stringLocation, indexBegin, indexEnd, ltr));
        }

        static Ref<ComplexTextRun> create(const Vector<FloatSize>& advances, const Vector<FloatPoint>& origins, const Vector<Glyph>& glyphs, const Vector<unsigned>& stringIndices, FloatSize initialAdvance, const Font& font, std::span<const UChar> characters, unsigned stringLocation, unsigned indexBegin, unsigned indexEnd, bool ltr)
        {
            return adoptRef(*new ComplexTextRun(advances, origins, glyphs, stringIndices, initialAdvance, font, characters, stringLocation, indexBegin, indexEnd, ltr));
        }

        unsigned glyphCount() const { return m_glyphCount; }
        const Font& font() const { return m_font; }
        std::span<const UChar> characters() const { return m_characters; }
        unsigned stringLocation() const { return m_stringLocation; }
        size_t stringLength() const { return m_characters.size(); }
        ALWAYS_INLINE unsigned indexAt(unsigned) const;
        unsigned indexBegin() const { return m_indexBegin; }
        unsigned indexEnd() const { return m_indexEnd; }
        unsigned endOffsetAt(unsigned i) const { ASSERT(!m_isMonotonic); return m_glyphEndOffsets[i]; }
        std::span<const CGGlyph> glyphs() const { return m_glyphs.span(); }

        void growInitialAdvanceHorizontally(float delta) { m_initialAdvance.expand(delta, 0); }
        FloatSize initialAdvance() const { return m_initialAdvance; }
        std::span<const FloatSize> baseAdvances() const { return m_baseAdvances.span(); }
        std::span<const FloatPoint> glyphOrigins() const { return m_glyphOrigins.size() == glyphCount() ? m_glyphOrigins.span() : std::span<const FloatPoint> { }; }
        bool isLTR() const { return m_isLTR; }
        bool isMonotonic() const { return m_isMonotonic; }
        void setIsNonMonotonic();
        float textAutospaceSize() const { return m_textAutospaceSize; }

    private:
        ComplexTextRun(CTRunRef, const Font&, std::span<const UChar> characters, unsigned stringLocation, unsigned indexBegin, unsigned indexEnd);
        ComplexTextRun(hb_buffer_t*, const Font&, std::span<const UChar> characters, unsigned stringLocation, unsigned indexBegin, unsigned indexEnd);
        ComplexTextRun(const Font&, std::span<const UChar> characters, unsigned stringLocation, unsigned indexBegin, unsigned indexEnd, bool ltr);
        WEBCORE_EXPORT ComplexTextRun(const Vector<FloatSize>& advances, const Vector<FloatPoint>& origins, const Vector<Glyph>& glyphs, const Vector<unsigned>& stringIndices, FloatSize initialAdvance, const Font&, std::span<const UChar> characters, unsigned stringLocation, unsigned indexBegin, unsigned indexEnd, bool ltr);

        using BaseAdvancesVector = Vector<FloatSize, 64>;
        using GlyphVector = Vector<CGGlyph, 64>;
        using CoreTextIndicesVector = Vector<unsigned, 64>;

        BaseAdvancesVector m_baseAdvances;
        Vector<FloatPoint, 64> m_glyphOrigins;
        GlyphVector m_glyphs;
        Vector<unsigned, 64> m_glyphEndOffsets;
        CoreTextIndicesVector m_coreTextIndices;
        FloatSize m_initialAdvance;
        const Font& m_font;
        std::span<const UChar> m_characters;
        unsigned m_indexBegin;
        unsigned m_indexEnd;
        unsigned m_glyphCount;
        unsigned m_stringLocation;
        bool m_isLTR;
        bool m_isMonotonic { true };
        float m_textAutospaceSize { 0 };
    };
private:
    ComplexTextController(const TextRun&, const FontCascade&);

    void computeExpansionOpportunity();
    void finishConstruction();
    
    static unsigned stringBegin(const ComplexTextRun& run) { return run.stringLocation() + run.indexBegin(); }
    static unsigned stringEnd(const ComplexTextRun& run) { return run.stringLocation() + run.indexEnd(); }

    void collectComplexTextRuns();

    void collectComplexTextRunsForCharacters(std::span<const UChar>, unsigned stringLocation, const Font*);
    void adjustGlyphsAndAdvances();

    unsigned indexOfCurrentRun(unsigned& leftmostGlyph);
    unsigned incrementCurrentRun(unsigned& leftmostGlyph);

    float runWidthSoFarFraction(unsigned glyphStartOffset, unsigned glyphEndOffset, unsigned oldCharacterInCurrentGlyph, GlyphIterationStyle) const;

    FloatPoint glyphOrigin(unsigned index) const { return index < m_glyphOrigins.size() ? m_glyphOrigins[index] : FloatPoint(); }

    void advanceByCombiningCharacterSequence(const WTF::CachedTextBreakIterator& graphemeClusterIterator, unsigned& location, char32_t& baseCharacter);

    Vector<FloatSize, 256> m_adjustedBaseAdvances;
    Vector<FloatPoint, 256> m_glyphOrigins;
    Vector<CGGlyph, 256> m_adjustedGlyphs;
    Vector<float, 256> m_textAutoSpaceSpacings;

    Vector<UChar, 256> m_smallCapsBuffer;

    // There is a 3-level hierarchy here. At the top, we are interested in m_run.string(). We partition that string
    // into Lines, each of which is a sequence of characters which should use the same Font. Core Text then partitions
    // the Line into ComplexTextRuns.
    // ComplexTextRun::stringLocation() and ComplexTextRun::stringLength() refer to the offset and length of the Line
    // relative to m_run.string(). ComplexTextRun::indexAt() returns to the offset of a codepoint relative to
    // its Line. ComplexTextRun::glyphs() and ComplexTextRun::advances() refer to glyphs relative to the ComplexTextRun.
    // The length of the entire TextRun is m_run.length()
    Vector<RefPtr<ComplexTextRun>, 16> m_complexTextRuns;

    // The initial capacity of these vectors was selected as being the smallest power of two greater than
    // the average (3.5) plus one standard deviation (7.5) of nonzero sizes used on Arabic Wikipedia.
    Vector<unsigned, 16> m_runIndices;
    Vector<unsigned, 16> m_glyphCountFromStartToIndex;

#if PLATFORM(COCOA)
    Vector<RetainPtr<CTLineRef>> m_coreTextLines;
#endif

    Vector<String> m_stringsFor8BitRuns;

    SingleThreadWeakHashSet<const Font>* m_fallbackFonts { nullptr };

    const FontCascade& m_font;
    const TextRun& m_run;

    unsigned m_currentCharacter { 0 };
    unsigned m_end { 0 };

    FloatSize m_totalAdvance;
    float m_runWidthSoFar { 0 };
    unsigned m_numGlyphsSoFar { 0 };
    unsigned m_currentRun { 0 };
    unsigned m_glyphInCurrentRun { 0 };
    unsigned m_characterInCurrentGlyph { 0 };
    float m_expansion { 0 };
    float m_expansionPerOpportunity { 0 };

    float m_minGlyphBoundingBoxX { std::numeric_limits<float>::max() };
    float m_maxGlyphBoundingBoxX { std::numeric_limits<float>::lowest() };
    float m_minGlyphBoundingBoxY { std::numeric_limits<float>::max() };
    float m_maxGlyphBoundingBoxY { std::numeric_limits<float>::min() };

    bool m_isLTROnly { true };
    bool m_mayUseNaturalWritingDirection { false };
    bool m_forTextEmphasis { false };
    TextSpacing::SpacingState m_textSpacingState;
};

} // namespace WebCore
