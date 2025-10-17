/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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


#include "InlineDisplayContent.h"
#include <wtf/HashMap.h>
#include <wtf/IteratorRange.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
namespace LayoutIntegration {
struct InlineContent;
}
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::LayoutIntegration::InlineContent> : std::true_type { };
}

namespace WebCore {

class RenderBlockFlow;
class RenderObject;
struct SVGTextFragment;

namespace Layout {
class Box;
}

namespace InlineDisplay {
struct Box;
class Line;
} 

namespace LayoutIntegration {

class LineLayout;

struct InlineContent : public CanMakeWeakPtr<InlineContent> {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    InlineContent(const LineLayout&);

    InlineDisplay::Content& displayContent() { return m_displayContent; }
    const InlineDisplay::Content& displayContent() const { return m_displayContent; }

    float clearGapBeforeFirstLine { 0 };
    float clearGapAfterLastLine { 0 };
    float firstLinePaginationOffset { 0 };

    bool isPaginated { false };
    bool hasMultilinePaintOverlap { false };

    bool hasContent() const;

    bool hasVisualOverflow() const { return m_hasVisualOverflow; }
    void setHasVisualOverflow() { m_hasVisualOverflow = true; }
    
    const InlineDisplay::Line& lineForBox(const InlineDisplay::Box& box) const { return displayContent().lines[box.lineIndex()]; }

    IteratorRange<const InlineDisplay::Box*> boxesForRect(const LayoutRect&) const;

    void shrinkToFit();

    const LineLayout& lineLayout() const { return *m_lineLayout; }
    const RenderBlockFlow& formattingContextRoot() const;

    size_t indexForBox(const InlineDisplay::Box&) const;

    const InlineDisplay::Box* firstBoxForLayoutBox(const Layout::Box&) const;
    template<typename Function> void traverseNonRootInlineBoxes(const Layout::Box&, Function&&);

    std::optional<size_t> firstBoxIndexForLayoutBox(const Layout::Box&) const;
    const Vector<size_t>& nonRootInlineBoxIndexesForLayoutBox(const Layout::Box&) const;

    const Vector<SVGTextFragment>& svgTextFragments(size_t boxIndex) const;
    Vector<Vector<SVGTextFragment>>& svgTextFragmentsForBoxes() { return m_svgTextFragmentsForBoxes; }

    void releaseCaches();

private:
    CheckedPtr<const LineLayout> m_lineLayout;

    InlineDisplay::Content m_displayContent;
    using FirstBoxIndexCache = UncheckedKeyHashMap<CheckedRef<const Layout::Box>, size_t>;
    mutable std::unique_ptr<FirstBoxIndexCache> m_firstBoxIndexCache;

    using InlineBoxIndexCache = UncheckedKeyHashMap<CheckedRef<const Layout::Box>, Vector<size_t>>;
    mutable std::unique_ptr<InlineBoxIndexCache> m_inlineBoxIndexCache;
    bool m_hasVisualOverflow { false };

    Vector<Vector<SVGTextFragment>> m_svgTextFragmentsForBoxes;
};

template<typename Function> void InlineContent::traverseNonRootInlineBoxes(const Layout::Box& layoutBox, Function&& function)
{
    for (auto index : nonRootInlineBoxIndexesForLayoutBox(layoutBox))
        function(displayContent().boxes[index]);
}

}
}

