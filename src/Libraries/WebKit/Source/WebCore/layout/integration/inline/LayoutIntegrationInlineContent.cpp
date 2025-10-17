/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
#include "LayoutIntegrationInlineContent.h"

#include "InlineIteratorBox.h"
#include "LayoutIntegrationLineLayout.h"
#include "RenderStyleInlines.h"
#include "SVGTextFragment.h"
#include "TextPainter.h"

namespace WebCore {
namespace LayoutIntegration {

InlineContent::InlineContent(const LineLayout& lineLayout)
    : m_lineLayout(lineLayout)
{
}

bool InlineContent::hasContent() const
{
    ASSERT(m_displayContent.boxes.isEmpty() || m_displayContent.boxes[0].isRootInlineBox());
    return m_displayContent.boxes.size() > 1;
}

IteratorRange<const InlineDisplay::Box*> InlineContent::boxesForRect(const LayoutRect& rect) const
{
    if (m_displayContent.boxes.isEmpty())
        return { nullptr, nullptr };

    auto& lines = m_displayContent.lines;
    auto& boxes = m_displayContent.boxes;

    // FIXME: Do the flips.
    if (formattingContextRoot().writingMode().isBlockFlipped())
        return { boxes.begin(), boxes.end() };

    if (lines.first().inkOverflow().maxY() > rect.y() && lines.last().inkOverflow().y() < rect.maxY())
        return { boxes.begin(), boxes.end() };

    // The optimization below relies on line paint bounds not exeeding those of the neighboring lines
    if (hasMultilinePaintOverlap)
        return { boxes.begin(), boxes.end() };

    auto height = lines.last().lineBoxBottom() - lines.first().lineBoxTop();
    auto averageLineHeight = height / lines.size();

    auto approximateLine = [&](LayoutUnit y) {
        y = std::max(y, 0_lu);
        return std::min(static_cast<size_t>(y / averageLineHeight), lines.size() - 1);
    };

    auto startLine = approximateLine(rect.y());
    for (; startLine; --startLine) {
        if (lines[startLine - 1].inkOverflow().maxY() < rect.y())
            break;
    }

    auto endLine = approximateLine(rect.maxY());
    for (; endLine < lines.size() - 1; ++endLine) {
        if (lines[endLine + 1].inkOverflow().y() > rect.maxY())
            break;
    }

    auto firstBox = lines[startLine].firstBoxIndex();
    auto lastBox = lines[endLine].firstBoxIndex() + lines[endLine].boxCount();

    auto boxSpan = boxes.subspan(firstBox, lastBox - firstBox);
    return { std::to_address(boxSpan.begin()), std::to_address(boxSpan.end()) };
}

const RenderBlockFlow& InlineContent::formattingContextRoot() const
{
    return lineLayout().flow();
}

size_t InlineContent::indexForBox(const InlineDisplay::Box& box) const
{
    auto index = static_cast<size_t>(&box - m_displayContent.boxes.begin());
    RELEASE_ASSERT(index < m_displayContent.boxes.size());
    return index;
}

const InlineDisplay::Box* InlineContent::firstBoxForLayoutBox(const Layout::Box& layoutBox) const
{
    auto index = firstBoxIndexForLayoutBox(layoutBox);
    return index ? &m_displayContent.boxes[*index] : nullptr;
}

std::optional<size_t> InlineContent::firstBoxIndexForLayoutBox(const Layout::Box& layoutBox) const
{
    constexpr auto cacheThreshold = 16;
    auto& boxes = m_displayContent.boxes;

    if (boxes.size() < cacheThreshold) {
        for (size_t i = 0; i < boxes.size(); ++i) {
            auto& box = boxes[i];
            if (&box.layoutBox() == &layoutBox)
                return i;
        }
        return { };
    }
    
    if (!m_firstBoxIndexCache) {
        m_firstBoxIndexCache = makeUnique<FirstBoxIndexCache>();
        for (size_t i = 0; i < boxes.size(); ++i) {
            auto& box = boxes[i];
            if (box.isRootInlineBox())
                continue;
            m_firstBoxIndexCache->add(box.layoutBox(), i);
        }
    }

    auto it = m_firstBoxIndexCache->find(layoutBox);
    if (it == m_firstBoxIndexCache->end())
        return { };

    return it->value;
}

const Vector<size_t>& InlineContent::nonRootInlineBoxIndexesForLayoutBox(const Layout::Box& layoutBox) const
{
    ASSERT(layoutBox.isElementBox());
    auto& boxes = m_displayContent.boxes;

    if (!m_inlineBoxIndexCache) {
        m_inlineBoxIndexCache = makeUnique<InlineBoxIndexCache>();
        for (size_t i = 0; i < boxes.size(); ++i) {
            auto& box = boxes[i];
            if (!box.isNonRootInlineBox())
                continue;
            m_inlineBoxIndexCache->ensure(box.layoutBox(), [&] {
                return Vector<size_t> { };
            }).iterator->value.append(i);
        }
        for (auto entry : *m_inlineBoxIndexCache)
            entry.value.shrinkToFit();
    }

    auto it = m_inlineBoxIndexCache->find(layoutBox);
    if (it == m_inlineBoxIndexCache->end()) {
        static NeverDestroyed<Vector<size_t>> emptyVector;
        return emptyVector.get();
    }

    return it->value;
}

const Vector<SVGTextFragment>& InlineContent::svgTextFragments(size_t index) const
{
    RELEASE_ASSERT(m_svgTextFragmentsForBoxes.size() > index);
    return m_svgTextFragmentsForBoxes[index];
}

void InlineContent::releaseCaches()
{
    m_firstBoxIndexCache = { };
    m_inlineBoxIndexCache = { };
}

void InlineContent::shrinkToFit()
{
    m_displayContent.boxes.shrinkToFit();
    m_displayContent.lines.shrinkToFit();
}

}
}
