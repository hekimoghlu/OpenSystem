/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 31, 2024.
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
#include "InlineIteratorInlineBox.h"

#include "LayoutIntegrationLineLayout.h"
#include "RenderBlockFlow.h"
#include "RenderInline.h"
#include "RenderStyleInlines.h"

namespace WebCore {
namespace InlineIterator {

InlineBox::InlineBox(PathVariant&& path)
    : Box(WTFMove(path))
{
}

RectEdges<bool> InlineBox::closedEdges() const
{
    // FIXME: Layout knows the answer to this question so we should consult it.
    RectEdges<bool> closedEdges { true };
    if (style().boxDecorationBreak() == BoxDecorationBreak::Clone)
        return closedEdges;
    auto writingMode = style().writingMode();
    bool isFirst = !nextInlineBoxLineLeftward() && !renderer().isContinuation();
    bool isLast = !nextInlineBoxLineRightward() && !renderer().continuation();
    closedEdges.setStart(isFirst, writingMode);
    closedEdges.setEnd(isLast, writingMode);
    return closedEdges;
};

InlineBoxIterator InlineBox::nextInlineBoxLineRightward() const
{
    return InlineBoxIterator(*this).traverseInlineBoxLineRightward();
}

InlineBoxIterator InlineBox::nextInlineBoxLineLeftward() const
{
    return InlineBoxIterator(*this).traverseInlineBoxLineLeftward();
}

LeafBoxIterator InlineBox::firstLeafBox() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) -> LeafBoxIterator {
        return { path.firstLeafBoxForInlineBox() };
    });
}

LeafBoxIterator InlineBox::lastLeafBox() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) -> LeafBoxIterator {
        return { path.lastLeafBoxForInlineBox() };
    });
}

LeafBoxIterator InlineBox::endLeafBox() const
{
    if (auto last = lastLeafBox())
        return last->nextLineRightwardOnLine();
    return { };
}

IteratorRange<BoxIterator> InlineBox::descendants() const
{
    BoxIterator begin(*this);
    begin.traverseLineRightwardOnLine();

    BoxIterator end(*this);
    end.traverseLineRightwardOnLineSkippingChildren();

    return { begin, end };
}

InlineBoxIterator::InlineBoxIterator(Box::PathVariant&& pathVariant)
    : BoxIterator(WTFMove(pathVariant))
{
}

InlineBoxIterator::InlineBoxIterator(const Box& box)
    : BoxIterator(box)
{
}

InlineBoxIterator& InlineBoxIterator::traverseInlineBoxLineRightward()
{
    WTF::switchOn(m_box.m_pathVariant, [](auto& path) {
        path.traverseNextInlineBox();
    });
    return *this;
}

InlineBoxIterator& InlineBoxIterator::traverseInlineBoxLineLeftward()
{
    WTF::switchOn(m_box.m_pathVariant, [](auto& path) {
        path.traversePreviousInlineBox();
    });
    return *this;
}

InlineBoxIterator lineLeftmostInlineBoxFor(const RenderInline& renderInline)
{
    if (auto* lineLayout = LayoutIntegration::LineLayout::containing(renderInline))
        return lineLayout->firstInlineBoxFor(renderInline);
    return { BoxLegacyPath { renderInline.firstLegacyInlineBox() } };
}

InlineBoxIterator firstRootInlineBoxFor(const RenderBlockFlow& block)
{
    if (auto* lineLayout = block.inlineLayout())
        return lineLayout->firstRootInlineBox();
    return { BoxLegacyPath { block.legacyRootBox() } };
}

InlineBoxIterator inlineBoxFor(const LegacyInlineFlowBox& legacyInlineFlowBox)
{
    return { BoxLegacyPath { &legacyInlineFlowBox } };
}

InlineBoxIterator inlineBoxFor(const LayoutIntegration::InlineContent& content, const InlineDisplay::Box& box)
{
    return inlineBoxFor(content, content.indexForBox(box));
}

InlineBoxIterator inlineBoxFor(const LayoutIntegration::InlineContent& content, size_t boxIndex)
{
    ASSERT(content.displayContent().boxes[boxIndex].isInlineBox());
    return { BoxModernPath { content, boxIndex } };
}

}
}
