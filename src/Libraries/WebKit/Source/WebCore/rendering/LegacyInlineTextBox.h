/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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

#include "GlyphDisplayListCacheRemoval.h"
#include "LegacyInlineBox.h"
#include "RenderText.h"
#include "TextBoxSelectableRange.h"
#include "TextRun.h"

namespace WebCore {

namespace InlineIterator {
class BoxLegacyPath;
}

class LegacyInlineTextBox : public LegacyInlineBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyInlineTextBox);
public:
    explicit LegacyInlineTextBox(RenderText& renderer)
        : LegacyInlineBox(renderer)
    {
    }

    virtual ~LegacyInlineTextBox();

    RenderText& renderer() const { return downcast<RenderText>(LegacyInlineBox::renderer()); }
    const RenderStyle& lineStyle() const { return isFirstLine() ? renderer().firstLineStyle() : renderer().style(); }

    LegacyInlineTextBox* prevTextBox() const { return m_prevTextBox; }
    LegacyInlineTextBox* nextTextBox() const { return m_nextTextBox; }
    void setNextTextBox(LegacyInlineTextBox* n) { m_nextTextBox = n; }
    void setPreviousTextBox(LegacyInlineTextBox* p) { m_prevTextBox = p; }

    bool hasTextContent() const;

    unsigned start() const { return m_start; }
    unsigned end() const { return m_start + m_len; }
    unsigned len() const { return m_len; }

    void setStart(unsigned start) { m_start = start; }
    void setLen(unsigned len) { m_len = len; }

    void offsetRun(int d) { ASSERT(!isDirty()); ASSERT(d > 0 || m_start >= static_cast<unsigned>(-d)); m_start += d; }

    TextBoxSelectableRange selectableRange() const;

    void markDirty(bool dirty = true) final;

    using LegacyInlineBox::setIsInGlyphDisplayListCache;

    LayoutUnit baselinePosition(FontBaseline) const final;
    LayoutUnit lineHeight() const final;

    LayoutRect logicalOverflowRect() const;
    void setLogicalOverflowRect(const LayoutRect&);
    LayoutUnit logicalTopVisualOverflow() const { return logicalOverflowRect().y(); }
    LayoutUnit logicalBottomVisualOverflow() const { return logicalOverflowRect().maxY(); }
    LayoutUnit logicalLeftVisualOverflow() const { return logicalOverflowRect().x(); }
    LayoutUnit logicalRightVisualOverflow() const { return logicalOverflowRect().maxX(); }

    virtual void dirtyOwnLineBoxes() { dirtyLineBoxes(); }

    void removeFromGlyphDisplayListCache();

#if ENABLE(TREE_DEBUGGING)
    void outputLineBox(WTF::TextStream&, bool mark, int depth) const final;
    ASCIILiteral boxName() const final;
#endif

private:
    LayoutUnit selectionTop() const;
    LayoutUnit selectionBottom() const;
    LayoutUnit selectionHeight() const;

public:
    virtual LayoutRect localSelectionRect(unsigned startPos, unsigned endPos) const;
    std::pair<unsigned, unsigned> selectionStartEnd() const;

private:
    void deleteLine() final;
    void extractLine() final;
    void attachLine() final;
    
public:
    RenderObject::HighlightState selectionState() const final;

public:
    bool isLineBreak() const final;

private:
    bool isInlineTextBox() const final { return true; }

public:
    int caretMinOffset() const final;
    int caretMaxOffset() const final;

private:
    float textPos() const; // returns the x position relative to the left start of the text line.

public:
    bool hasMarkers() const;

private:
    friend class InlineIterator::BoxLegacyPath;

    const FontCascade& lineFont() const;

    String text() const; // The effective text for the run.
    TextRun createTextRun() const;

    LegacyInlineTextBox* m_prevTextBox { nullptr }; // The previous box that also uses our RenderObject
    LegacyInlineTextBox* m_nextTextBox { nullptr }; // The next box that also uses our RenderObject

    unsigned m_start { 0 };
    unsigned m_len { 0 };
};

inline void LegacyInlineTextBox::removeFromGlyphDisplayListCache()
{
    if (isInGlyphDisplayListCache()) {
        removeBoxFromGlyphDisplayListCache(*this);
        setIsInGlyphDisplayListCache(false);
    }
}

LayoutRect snappedSelectionRect(const LayoutRect&, float logicalRight, WritingMode);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INLINE_BOX(LegacyInlineTextBox, isInlineTextBox())
