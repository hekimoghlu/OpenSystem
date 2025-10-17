/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#include "InlineLineBox.h"

#include "InlineFormattingUtils.h"
#include "InlineLevelBoxInlines.h"
#include "LayoutBoxGeometry.h"
#include "LayoutElementBox.h"
#include "RenderStyleInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Layout {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LineBox);

LineBox::LineBox(const Box& rootLayoutBox, InlineLayoutUnit contentLogicalLeft, InlineLayoutUnit contentLogicalWidth, size_t lineIndex, size_t nonSpanningInlineLevelBoxCount)
    : m_lineIndex(lineIndex)
    , m_rootInlineBox(InlineLevelBox::createRootInlineBox(rootLayoutBox, !lineIndex ? rootLayoutBox.firstLineStyle() : rootLayoutBox.style(), contentLogicalLeft, contentLogicalWidth))
{
    m_nonRootInlineLevelBoxList.reserveInitialCapacity(nonSpanningInlineLevelBoxCount);
    m_nonRootInlineLevelBoxMap.reserveInitialCapacity(nonSpanningInlineLevelBoxCount);
    m_rootInlineBox.setTextEmphasis(InlineFormattingUtils::textEmphasisForInlineBox(rootLayoutBox, downcast<ElementBox>(rootLayoutBox)));
}

void LineBox::addInlineLevelBox(InlineLevelBox&& inlineLevelBox)
{
    m_boxTypes.add(inlineLevelBox.type());
    m_nonRootInlineLevelBoxMap.set(&inlineLevelBox.layoutBox(), m_nonRootInlineLevelBoxList.size());
    m_nonRootInlineLevelBoxList.append(WTFMove(inlineLevelBox));
}

InlineRect LineBox::logicalRectForTextRun(const Line::Run& run) const
{
    ASSERT(run.isText() || run.isSoftLineBreak());
    auto* ancestorInlineBox = &parentInlineBox(run);
    ASSERT(ancestorInlineBox->isInlineBox());
    auto runlogicalTop = ancestorInlineBox->logicalTop() - ancestorInlineBox->inlineBoxContentOffsetForTextBoxTrim();
    InlineLayoutUnit logicalHeight = ancestorInlineBox->primarymetricsOfPrimaryFont().intHeight();

    while (ancestorInlineBox != &m_rootInlineBox && !ancestorInlineBox->hasLineBoxRelativeAlignment()) {
        ancestorInlineBox = &parentInlineBox(*ancestorInlineBox);
        ASSERT(ancestorInlineBox->isInlineBox());
        runlogicalTop += ancestorInlineBox->logicalTop();
    }
    return { runlogicalTop, m_rootInlineBox.logicalLeft() + run.logicalLeft(), run.logicalWidth(), logicalHeight };
}

InlineRect LineBox::logicalRectForLineBreakBox(const Box& layoutBox) const
{
    ASSERT(layoutBox.isLineBreakBox());
    return logicalRectForInlineLevelBox(layoutBox);
}

InlineLayoutUnit LineBox::inlineLevelBoxAbsoluteTop(const InlineLevelBox& inlineLevelBox) const
{
    // Inline level boxes are relative to their parent unless the vertical alignment makes them relative to the line box (e.g. top, bottom).
    auto top = inlineLevelBox.logicalTop();
    if (inlineLevelBox.isRootInlineBox() || inlineLevelBox.hasLineBoxRelativeAlignment())
        return top;

    // Fast path for inline level boxes on the root inline box (e.g <div><img></div>).
    if (&inlineLevelBox.layoutBox().parent() == &m_rootInlineBox.layoutBox())
        return top + m_rootInlineBox.logicalTop();

    // Nested inline content e.g <div><span><img></span></div>
    auto* ancestorInlineBox = &inlineLevelBox;
    while (ancestorInlineBox != &m_rootInlineBox && !ancestorInlineBox->hasLineBoxRelativeAlignment()) {
        ancestorInlineBox = &parentInlineBox(*ancestorInlineBox);
        ASSERT(ancestorInlineBox->isInlineBox());
        top += ancestorInlineBox->logicalTop();
    }
    return top;
}

InlineRect LineBox::logicalRectForInlineLevelBox(const Box& layoutBox) const
{
    ASSERT(layoutBox.isInlineLevelBox() || layoutBox.isLineBreakBox());
    auto* inlineBox = inlineLevelBoxFor(layoutBox);
    if (!inlineBox) {
        ASSERT_NOT_REACHED();
        return { };
    }
    auto inlineBoxLogicalRect = inlineBox->logicalRect();
    return InlineRect { inlineLevelBoxAbsoluteTop(*inlineBox), inlineBoxLogicalRect.left(), inlineBoxLogicalRect.width(), inlineBoxLogicalRect.height() };
}

InlineRect LineBox::logicalBorderBoxForAtomicInlineBox(const Box& layoutBox, const BoxGeometry& boxGeometry) const
{
    ASSERT(layoutBox.isAtomicInlineBox());
    auto logicalRect = logicalRectForInlineLevelBox(layoutBox);
    // Inline level boxes use their margin box for vertical alignment. Let's covert them to border boxes.
    logicalRect.moveVertically(boxGeometry.marginBefore());
    auto verticalMargin = boxGeometry.marginBefore() + boxGeometry.marginAfter();
    logicalRect.expandVertically(-verticalMargin);

    return logicalRect;
}

InlineRect LineBox::logicalBorderBoxForInlineBox(const Box& layoutBox, const BoxGeometry& boxGeometry) const
{
    auto logicalRect = logicalRectForInlineLevelBox(layoutBox);
    // This logical rect is as tall as the "text" content is. Let's adjust with vertical border and padding.
    logicalRect.expandVertically(boxGeometry.verticalBorderAndPadding());
    logicalRect.moveVertically(-boxGeometry.borderAndPaddingBefore());
    return logicalRect;
}

} // namespace Layout
} // namespace WebCore

