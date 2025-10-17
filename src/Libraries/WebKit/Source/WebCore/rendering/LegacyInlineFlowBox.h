/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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

#include "LegacyInlineBox.h"
#include "RenderOverflow.h"
#include "ShadowData.h"

namespace WebCore {

class HitTestRequest;
class HitTestResult;
class LegacyInlineTextBox;
class RenderLineBoxList;
class Font;

struct GlyphOverflow;

typedef UncheckedKeyHashMap<const LegacyInlineTextBox*, std::pair<Vector<SingleThreadWeakPtr<const Font>>, GlyphOverflow>> GlyphOverflowAndFallbackFontsMap;

class LegacyInlineFlowBox : public LegacyInlineBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyInlineFlowBox);
public:
    explicit LegacyInlineFlowBox(RenderBoxModelObject& renderer)
        : LegacyInlineBox(renderer)
    {
    }

#if !ASSERT_WITH_SECURITY_IMPLICATION_DISABLED
    virtual ~LegacyInlineFlowBox();
#endif

#if ENABLE(TREE_DEBUGGING)
    void outputLineTreeAndMark(WTF::TextStream&, const LegacyInlineBox* markedBox, int depth) const override;
    ASCIILiteral boxName() const override;
#endif

    RenderBoxModelObject& renderer() const { return downcast<RenderBoxModelObject>(LegacyInlineBox::renderer()); }
    const RenderStyle& lineStyle() const { return isFirstLine() ? renderer().firstLineStyle() : renderer().style(); }

    LegacyInlineFlowBox* prevLineBox() const { return m_prevLineBox; }
    LegacyInlineFlowBox* nextLineBox() const { return m_nextLineBox; }
    void setNextLineBox(LegacyInlineFlowBox* n) { m_nextLineBox = n; }
    void setPreviousLineBox(LegacyInlineFlowBox* p) { m_prevLineBox = p; }

    LegacyInlineBox* firstChild() const { checkConsistency(); return m_firstChild; }
    LegacyInlineBox* lastChild() const { checkConsistency(); return m_lastChild; }

    bool isLeaf() const final { return false; }
    
    LegacyInlineBox* firstLeafDescendant() const;
    LegacyInlineBox* lastLeafDescendant() const;

    void setConstructed() final
    {
        LegacyInlineBox::setConstructed();
        for (auto* child = firstChild(); child; child = child->nextOnLine())
            child->setConstructed();
    }

    void addToLine(LegacyInlineBox* child);
    void deleteLine() final;
    void extractLine() final;
    void attachLine() final;
    void adjustPosition(float dx, float dy) override;

    virtual void extractLineBoxFromRenderObject();
    virtual void attachLineBoxToRenderObject();
    virtual void removeLineBoxFromRenderObject();

    void computeOverflow(LayoutUnit lineTop, LayoutUnit lineBottom, GlyphOverflowAndFallbackFontsMap&);
    
    void removeChild(LegacyInlineBox* child);

    RenderObject::HighlightState selectionState() const override;

    bool hasTextChildren() const { return m_hasTextChildren; }
    bool hasTextDescendants() const { return m_hasTextDescendants; }
    void setHasTextChildren() { m_hasTextChildren = true; setHasTextDescendants(); }
    void setHasTextDescendants() { m_hasTextDescendants = true; }
    
    void checkConsistency() const;
    void setHasBadChildList();

    LayoutRect visualOverflowRect(LayoutUnit lineTop, LayoutUnit lineBottom) const
    { 
        return m_overflow ? m_overflow->visualOverflowRect() : enclosingLayoutRect(frameRectIncludingLineHeight(lineTop, lineBottom));
    }
    LayoutUnit logicalLeftVisualOverflow() const { return m_overflow ? (isHorizontal() ? m_overflow->visualOverflowRect().x() : m_overflow->visualOverflowRect().y()) : LayoutUnit(logicalLeft()); }
    LayoutUnit logicalRightVisualOverflow() const { return m_overflow ? (isHorizontal() ? m_overflow->visualOverflowRect().maxX() : m_overflow->visualOverflowRect().maxY()) : LayoutUnit(ceilf(logicalRight())); }
    LayoutUnit logicalTopVisualOverflow(LayoutUnit lineTop) const
    {
        if (m_overflow)
            return isHorizontal() ? m_overflow->visualOverflowRect().y() : m_overflow->visualOverflowRect().x();
        return lineTop;
    }
    LayoutUnit logicalBottomVisualOverflow(LayoutUnit lineBottom) const
    {
        if (m_overflow)
            return isHorizontal() ? m_overflow->visualOverflowRect().maxY() : m_overflow->visualOverflowRect().maxX();
        return lineBottom;
    }
    LayoutRect logicalVisualOverflowRect(LayoutUnit lineTop, LayoutUnit lineBottom) const
    {
        LayoutRect result = visualOverflowRect(lineTop, lineBottom);
        if (!renderer().isHorizontalWritingMode())
            result = result.transposedRect();
        return result;
    }

    void setOverflowFromLogicalRects(const LayoutRect& logicalVisualOverflow, LayoutUnit lineTop, LayoutUnit lineBottom);
    void setVisualOverflow(const LayoutRect&, LayoutUnit lineTop, LayoutUnit lineBottom);

    FloatRect frameRectIncludingLineHeight(LayoutUnit lineTop, LayoutUnit lineBottom) const
    {
        if (isHorizontal())
            return FloatRect(x(), lineTop, width(), lineBottom - lineTop);
        return FloatRect(lineTop, y(), lineBottom - lineTop, height());
    }
    
    FloatRect logicalFrameRectIncludingLineHeight(LayoutUnit lineTop, LayoutUnit lineBottom) const
    {
        return FloatRect(logicalLeft(), lineTop, logicalWidth(), lineBottom - lineTop);
    }

    bool hasSelfPaintInlineBox() const { return m_hasSelfPaintInlineBox; }

private:
    bool isInlineFlowBox() const final { return true; }
    void boxModelObject() const = delete;

    void addTextBoxVisualOverflow(LegacyInlineTextBox&, GlyphOverflowAndFallbackFontsMap&, LayoutRect& logicalVisualOverflow);

private:
    unsigned m_hasTextChildren : 1 { false };
    unsigned m_hasTextDescendants : 1 { false };

protected:
    // The following members are only used by RootInlineBox but moved here to keep the bits packed.

    // Whether or not this line uses alphabetic or ideographic baselines by default.
    unsigned m_baselineType : 1 { AlphabeticBaseline }; // FontBaseline

    unsigned m_lineBreakBidiStatusEor : 5; // UCharDirection
    unsigned m_lineBreakBidiStatusLastStrong : 5; // UCharDirection
    unsigned m_lineBreakBidiStatusLast : 5; // UCharDirection

    unsigned m_hasSelfPaintInlineBox : 1 { false };

    // End of RootInlineBox-specific members.

#if !ASSERT_WITH_SECURITY_IMPLICATION_DISABLED
private:
    unsigned m_hasBadChildList : 1 { false };
#endif

protected:
    RefPtr<RenderOverflow> m_overflow;

    LegacyInlineBox* m_firstChild { nullptr };
    LegacyInlineBox* m_lastChild { nullptr };

    LegacyInlineFlowBox* m_prevLineBox { nullptr }; // The previous box that also uses our RenderObject
    LegacyInlineFlowBox* m_nextLineBox { nullptr }; // The next box that also uses our RenderObject
};

#ifdef NDEBUG

inline void LegacyInlineFlowBox::checkConsistency() const
{
}

#endif

#if ASSERT_WITH_SECURITY_IMPLICATION_DISABLED

inline void LegacyInlineFlowBox::setHasBadChildList()
{
}

#endif

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INLINE_BOX(LegacyInlineFlowBox, isInlineFlowBox())

#if ENABLE(TREE_DEBUGGING)
// Outside the WebCore namespace for ease of invocation from the debugger.
void showTree(const WebCore::LegacyInlineFlowBox*);
#endif
