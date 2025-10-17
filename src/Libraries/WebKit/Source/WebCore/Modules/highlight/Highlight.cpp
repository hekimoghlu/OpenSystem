/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#include "Highlight.h"

#include "IDLTypes.h"
#include "JSDOMSetLike.h"
#include "JSStaticRange.h"
#include "NodeTraversal.h"
#include "Range.h"
#include "RenderBlockFlow.h"
#include "StaticRange.h"

namespace WebCore {

void Highlight::repaintRange(const AbstractRange& range)
{
    auto sortedRange = makeSimpleRange(range);
    if (is_gt(treeOrder<ComposedTree>(sortedRange.start, sortedRange.end)))
        std::swap(sortedRange.start, sortedRange.end);
    for (Ref node : intersectingNodes(sortedRange)) {
        if (auto renderer = node->renderer())
            renderer->repaint();
    }
}

Ref<Highlight> Highlight::create(FixedVector<std::reference_wrapper<WebCore::AbstractRange>>&& initialRanges)
{
    return adoptRef(*new Highlight(WTFMove(initialRanges)));
}

Highlight::Highlight(FixedVector<std::reference_wrapper<WebCore::AbstractRange>>&& initialRanges)
{
    m_highlightRanges = WTF::map(initialRanges, [&](auto&& range) {
        repaintRange(range.get());
        return HighlightRange::create(range.get());
    });
}

void Highlight::initializeSetLike(DOMSetAdapter& set)
{
    for (auto& highlightRange : m_highlightRanges)
        set.add<IDLInterface<AbstractRange>>(highlightRange->range());
}

bool Highlight::removeFromSetLike(const AbstractRange& range)
{
    return m_highlightRanges.removeFirstMatching([&range](const Ref<HighlightRange>& current) {
        repaintRange(range);
        return &current->range() == &range;
    });
}

void Highlight::clearFromSetLike()
{
    for (auto& highlightRange : m_highlightRanges)
        repaintRange(highlightRange->range());
    m_highlightRanges.clear();
}

bool Highlight::addToSetLike(AbstractRange& range)
{
    auto index = m_highlightRanges.findIf([&range](const Ref<HighlightRange>& current) {
        return &current->range() == &range;
    });
    if (index == notFound) {
        repaintRange(range);
        m_highlightRanges.append(HighlightRange::create(range));
        return true;
    }
    // Move to last since SetLike is an ordered set.
    m_highlightRanges.append(WTFMove(m_highlightRanges[index]));
    m_highlightRanges.remove(index);
    return false;
}

void Highlight::repaint()
{
    for (auto& highlightRange : m_highlightRanges)
        repaintRange(highlightRange->range());
}

void Highlight::setPriority(int priority)
{
    if (m_priority == priority)
        return;
    m_priority = priority;
    repaint();
}

} // namespace WebCore
