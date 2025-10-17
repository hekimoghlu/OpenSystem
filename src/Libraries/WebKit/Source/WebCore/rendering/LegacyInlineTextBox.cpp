/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#include "LegacyInlineTextBox.h"

#include "BreakLines.h"
#include "CompositionHighlight.h"
#include "DashArray.h"
#include "Document.h"
#include "DocumentMarkerController.h"
#include "Editor.h"
#include "ElementRuleCollector.h"
#include "EventRegion.h"
#include "FloatRoundedRect.h"
#include "GlyphDisplayListCache.h"
#include "GraphicsContext.h"
#include "HitTestResult.h"
#include "ImageBuffer.h"
#include "InlineIteratorBoxInlines.h"
#include "InlineIteratorTextBox.h"
#include "InlineIteratorTextBoxInlines.h"
#include "InlineTextBoxStyle.h"
#include "LocalFrame.h"
#include "Page.h"
#include "RenderBlock.h"
#include "RenderCombineText.h"
#include "RenderElementInlines.h"
#include "RenderHighlight.h"
#include "RenderLineBreak.h"
#include "RenderStyleInlines.h"
#include "RenderTheme.h"
#include "RenderView.h"
#include "RenderedDocumentMarker.h"
#include "Settings.h"
#include "StyledMarkedText.h"
#include "Text.h"
#include "TextBoxPainter.h"
#include "TextBoxSelectableRange.h"
#include <stdio.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/CString.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyInlineTextBox);

struct SameSizeAsLegacyInlineTextBox : public LegacyInlineBox {
    void* pointers[2];
    unsigned variables[2];
};

static_assert(sizeof(LegacyInlineTextBox) == sizeof(SameSizeAsLegacyInlineTextBox), "LegacyInlineTextBox should stay small");

typedef UncheckedKeyHashMap<const LegacyInlineTextBox*, LayoutRect> LegacyInlineTextBoxOverflowMap;
static LegacyInlineTextBoxOverflowMap* gTextBoxesWithOverflow;

LegacyInlineTextBox::~LegacyInlineTextBox()
{
    if (!knownToHaveNoOverflow() && gTextBoxesWithOverflow)
        gTextBoxesWithOverflow->remove(this);
    if (isInGlyphDisplayListCache())
        removeBoxFromGlyphDisplayListCache(*this);
}

bool LegacyInlineTextBox::hasTextContent() const
{
    return m_len;
}

void LegacyInlineTextBox::markDirty(bool dirty)
{
    if (dirty) {
        m_len = 0;
        m_start = 0;
    }
    LegacyInlineBox::markDirty(dirty);
}

LayoutRect LegacyInlineTextBox::logicalOverflowRect() const
{
    if (knownToHaveNoOverflow() || !gTextBoxesWithOverflow)
        return enclosingIntRect(logicalFrameRect());
    return gTextBoxesWithOverflow->get(this);
}

void LegacyInlineTextBox::setLogicalOverflowRect(const LayoutRect& rect)
{
    ASSERT(!knownToHaveNoOverflow());
    if (!gTextBoxesWithOverflow)
        gTextBoxesWithOverflow = new LegacyInlineTextBoxOverflowMap;
    gTextBoxesWithOverflow->add(this, rect);
}

LayoutUnit LegacyInlineTextBox::baselinePosition(FontBaseline baselineType) const
{
    if (!parent())
        return 0;
    if (&parent()->renderer() == renderer().parent())
        return parent()->baselinePosition(baselineType);
    return downcast<RenderBoxModelObject>(*renderer().parent()).baselinePosition(baselineType, isFirstLine(), isHorizontal() ? HorizontalLine : VerticalLine, PositionOnContainingLine);
}

LayoutUnit LegacyInlineTextBox::lineHeight() const
{
    if (!renderer().parent())
        return 0;
    if (&parent()->renderer() == renderer().parent())
        return parent()->lineHeight();
    return downcast<RenderBoxModelObject>(*renderer().parent()).lineHeight(isFirstLine(), isHorizontal() ? HorizontalLine : VerticalLine, PositionOnContainingLine);
}

LayoutUnit LegacyInlineTextBox::selectionTop() const
{
    return root().selectionTop();
}

LayoutUnit LegacyInlineTextBox::selectionBottom() const
{
    return root().selectionBottom();
}

LayoutUnit LegacyInlineTextBox::selectionHeight() const
{
    return root().selectionHeight();
}

RenderObject::HighlightState LegacyInlineTextBox::selectionState() const
{
    return renderer().view().selection().highlightStateForTextBox(renderer(), selectableRange());
}

const FontCascade& LegacyInlineTextBox::lineFont() const
{
    return lineStyle().fontCascade();
}

LayoutRect snappedSelectionRect(const LayoutRect& selectionRect, float logicalRight, WritingMode writingMode)
{
    auto snappedSelectionRect = enclosingIntRect(selectionRect);
    LayoutUnit logicalWidth = snappedSelectionRect.width();
    if (snappedSelectionRect.x() > logicalRight)
        logicalWidth = 0;
    else if (snappedSelectionRect.maxX() > logicalRight)
        logicalWidth = logicalRight - snappedSelectionRect.x();

    if (writingMode.isHorizontal()) {
        return {
            snappedSelectionRect.x(), selectionRect.y(),
            logicalWidth, selectionRect.height(),
        };
    }
    return {
        selectionRect.y(), snappedSelectionRect.x(),
        selectionRect.height(), logicalWidth,
    };
}

LayoutRect LegacyInlineTextBox::localSelectionRect(unsigned startPos, unsigned endPos) const
{
    auto [clampedStart, clampedEnd] = selectableRange().clamp(startPos, endPos);

    if (clampedStart >= clampedEnd && !(startPos == endPos && startPos >= start() && startPos <= (start() + len())))
        return { };

    TextRun textRun = createTextRun();
    auto writingMode = renderer().writingMode();
    auto width = LayoutUnit { logicalWidth() };

    LayoutRect selectionRect { 0, this->selectionTop(), width, this->selectionHeight() };
    // Avoid measuring the text when the entire line box is selected as an optimization.
    if (clampedStart || clampedEnd != textRun.length())
        lineFont().adjustSelectionRectForText(renderer().canUseSimplifiedTextMeasuring().value_or(false), textRun, selectionRect, clampedStart, clampedEnd);

    if (!writingMode.isLogicalLeftLineLeft())
        selectionRect.setX(width - selectionRect.x());
    selectionRect.move(logicalLeft(), 0);
    // FIXME: The computation of the snapped selection rect differs from the computation of this rect
    // in paintMarkedTextBackground(). See <https://bugs.webkit.org/show_bug.cgi?id=138913>.
    return snappedSelectionRect(selectionRect, logicalRight(), renderer().writingMode());
}

void LegacyInlineTextBox::deleteLine()
{
    renderer().removeTextBox(*this);
    delete this;
}

void LegacyInlineTextBox::extractLine()
{
    if (extracted())
        return;

    renderer().extractTextBox(*this);
}

void LegacyInlineTextBox::attachLine()
{
    if (!extracted())
        return;
    
    renderer().attachTextBox(*this);
}

bool LegacyInlineTextBox::isLineBreak() const
{
    return renderer().style().preserveNewline() && len() == 1 && renderer().text()[start()] == '\n';
}

TextBoxSelectableRange LegacyInlineTextBox::selectableRange() const
{
    // Fix up the offset if we are combined text because we manage these embellishments.
    // That is, they are not reflected in renderer().text(). We treat combined text as a single unit.
    return {
        m_start,
        m_len,
        0u,
        isLineBreak()
    };
}

std::pair<unsigned, unsigned> LegacyInlineTextBox::selectionStartEnd() const
{
    return renderer().view().selection().rangeForTextBox(renderer(), selectableRange());
}

bool LegacyInlineTextBox::hasMarkers() const
{
    return MarkedText::collectForDocumentMarkers(renderer(), selectableRange(), MarkedText::PaintPhase::Decoration).size();
}

int LegacyInlineTextBox::caretMinOffset() const
{
    return m_start;
}

int LegacyInlineTextBox::caretMaxOffset() const
{
    return m_start + m_len;
}

float LegacyInlineTextBox::textPos() const
{
    // When computing the width of a text run, RenderBlock::computeInlineDirectionPositionsForLine() doesn't include the actual offset
    // from the containing block edge in its measurement. textPos() should be consistent so the text are rendered in the same width.
    if (!logicalLeft())
        return 0;
    return logicalLeft() - root().logicalLeft();
}

TextRun LegacyInlineTextBox::createTextRun() const
{
    const auto& style = lineStyle();
    TextRun textRun { text(), textPos(), 0, ExpansionBehavior::forbidAll(), direction(), style.rtlOrdering() == Order::Visual, !renderer().canUseSimpleFontCodePath() };
    textRun.setTabSize(!style.collapseWhiteSpace(), style.tabSize());
    return textRun;
}

String LegacyInlineTextBox::text() const
{
    String result = renderer().text().substring(m_start, m_len);

    // This works because this replacement doesn't affect string indices. We're replacing a single Unicode code unit with another Unicode code unit.
    // How convenient.
    return RenderBlock::updateSecurityDiscCharacters(lineStyle(), WTFMove(result));
}

#if ENABLE(TREE_DEBUGGING)

ASCIILiteral LegacyInlineTextBox::boxName() const
{
    return "InlineTextBox"_s;
}

void LegacyInlineTextBox::outputLineBox(TextStream& stream, bool mark, int depth) const
{
    stream << "-------- "_s << (isDirty() ? "D"_s : "-"_s) << "-"_s;

    int printedCharacters = 0;
    if (mark) {
        stream << "*"_s;
        ++printedCharacters;
    }
    while (++printedCharacters <= depth * 2)
        stream << " "_s;

    String value = renderer().text();
    value = value.substring(start(), len());
    value = makeStringByReplacingAll(value, '\\', "\\\\"_s);
    value = makeStringByReplacingAll(value, '\n', "\\n"_s);
    stream << boxName() << " "_s << FloatRect(x(), y(), width(), height()) << " ("_s << this << ") renderer->("_s << &renderer() << ") run("_s << start() << ", "_s << start() + len() << ") \""_s << value.utf8().data() << "\""_s;
    stream.nextLine();
}

#endif

} // namespace WebCore
