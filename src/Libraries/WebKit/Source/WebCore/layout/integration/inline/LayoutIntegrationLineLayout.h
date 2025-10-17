/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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

#include "FloatRect.h"
#include "InlineDamage.h"
#include "InlineFormattingConstraints.h"
#include "InlineFormattingContext.h"
#include "InlineIteratorInlineBox.h"
#include "InlineIteratorLineBox.h"
#include "InlineIteratorTextBox.h"
#include "LayoutIntegrationBoxGeometryUpdater.h"
#include "LayoutIntegrationBoxTreeUpdater.h"
#include "LayoutPoint.h"
#include "LayoutState.h"
#include "RenderObjectEnums.h"
#include "SVGTextChunk.h"
#include <wtf/CheckedPtr.h>

namespace WebCore {

class HitTestLocation;
class HitTestRequest;
class HitTestResult;
class RenderBlockFlow;
class RenderBox;
class RenderBoxModelObject;
class RenderInline;
struct PaintInfo;

namespace Layout {
class InlineDamage;
}

namespace LayoutIntegration {

struct InlineContent;
struct LineAdjustment;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(LayoutIntegration_LineLayout);

class LineLayout final : public CanMakeCheckedPtr<LineLayout> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(LayoutIntegration_LineLayout);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LineLayout);
public:
    LineLayout(RenderBlockFlow&);
    ~LineLayout();

    static RenderBlockFlow* blockContainer(const RenderObject&);
    static LineLayout* containing(RenderObject&);
    static const LineLayout* containing(const RenderObject&);

    static bool canUseFor(const RenderBlockFlow&);
    static bool canUseForPreferredWidthComputation(const RenderBlockFlow&);
    static bool shouldInvalidateLineLayoutPathAfterContentChange(const RenderBlockFlow& parent, const RenderObject& rendererWithNewContent, const LineLayout&);
    static bool shouldInvalidateLineLayoutPathAfterTreeMutation(const RenderBlockFlow& parent, const RenderObject& renderer, const LineLayout&, bool isRemoval);

    void updateFormattingContexGeometries(LayoutUnit availableLogicalWidth);
    void updateOverflow();
    static void updateStyle(const RenderObject&);

    // Partial invalidation.
    bool insertedIntoTree(const RenderElement& parent, RenderObject& child);
    bool removedFromTree(const RenderElement& parent, RenderObject& child);
    bool updateTextContent(const RenderText&, size_t offset, int delta);
    bool rootStyleWillChange(const RenderBlockFlow&, const RenderStyle& newStyle);
    bool styleWillChange(const RenderElement&, const RenderStyle& newStyle, StyleDifference);
    bool boxContentWillChange(const RenderBox&);

    std::pair<LayoutUnit, LayoutUnit> computeIntrinsicWidthConstraints();

    std::optional<LayoutRect> layout();
    void paint(PaintInfo&, const LayoutPoint& paintOffset, const RenderInline* layerRenderer = nullptr);
    bool hitTest(const HitTestRequest&, HitTestResult&, const HitTestLocation&, const LayoutPoint& accumulatedOffset, HitTestAction, const RenderInline* layerRenderer = nullptr);
    void adjustForPagination();
    void shiftLinesBy(LayoutUnit blockShift);

    void collectOverflow();
    LayoutRect visualOverflowBoundingBoxRectFor(const RenderInline&) const;
    Vector<FloatRect> collectInlineBoxRects(const RenderInline&) const;

    LayoutUnit contentLogicalHeight() const;
    std::optional<LayoutUnit> clampedContentLogicalHeight() const;
    bool hasEllipsisInBlockDirectionOnLastFormattedLine() const;
    bool contains(const RenderElement& renderer) const;

    bool isPaginated() const;
    size_t lineCount() const;
    bool hasVisualOverflow() const;
    LayoutUnit firstLinePhysicalBaseline() const;
    LayoutUnit lastLinePhysicalBaseline() const;
    LayoutUnit lastLineLogicalBaseline() const;
    LayoutRect firstInlineBoxRect(const RenderInline&) const;
    LayoutRect enclosingBorderBoxRectFor(const RenderInline&) const;

    InlineIterator::TextBoxIterator textBoxesFor(const RenderText&) const;
    InlineIterator::LeafBoxIterator boxFor(const RenderElement&) const;
    InlineIterator::InlineBoxIterator firstInlineBoxFor(const RenderInline&) const;
    InlineIterator::InlineBoxIterator firstRootInlineBox() const;
    InlineIterator::LineBoxIterator firstLineBox() const;
    InlineIterator::LineBoxIterator lastLineBox() const;

    const RenderBlockFlow& flow() const { return downcast<RenderBlockFlow>(*m_rootLayoutBox->rendererForIntegration()); }
    RenderBlockFlow& flow() { return downcast<RenderBlockFlow>(*m_rootLayoutBox->rendererForIntegration()); }

    static void releaseCaches(RenderView&);

#if ENABLE(TREE_DEBUGGING)
    void outputLineTree(WTF::TextStream&, size_t depth) const;
#endif

    // This is temporary, required by partial bailout check.
    bool contentNeedsVisualReordering() const;
    bool isDamaged() const { return !!m_lineDamage; }
    const Layout::InlineDamage* damage() const { return m_lineDamage.get(); }
#ifndef NDEBUG
    bool hasDetachedContent() const { return m_lineDamage && m_lineDamage->hasDetachedContent(); }
#endif

    FloatRect applySVGTextFragments(SVGTextFragmentMap&&);

private:
    void preparePlacedFloats();
    FloatRect constructContent(const Layout::InlineLayoutState&, Layout::InlineLayoutResult&&);
    Vector<LineAdjustment> adjustContentForPagination(const Layout::BlockLayoutState&, bool isPartialLayout);
    void updateRenderTreePositions(const Vector<LineAdjustment>&, const Layout::InlineLayoutState&, bool didDiscardContent);

    InlineContent& ensureInlineContent();

    Layout::LayoutState& layoutState() { return *m_layoutState; }
    const Layout::LayoutState& layoutState() const { return *m_layoutState; }

    Layout::InlineDamage& ensureLineDamage();

    const Layout::ElementBox& rootLayoutBox() const { return *m_rootLayoutBox; }
    Layout::ElementBox& rootLayoutBox() { return *m_rootLayoutBox; }
    void clearInlineContent();
    void releaseCachesAndResetDamage();

    LayoutUnit physicalBaselineForLine(const InlineDisplay::Line&) const;
    
    CheckedPtr<Layout::ElementBox> m_rootLayoutBox;
    WeakPtr<Layout::LayoutState> m_layoutState;
    Layout::BlockFormattingState& m_blockFormattingState;
    Layout::InlineContentCache& m_inlineContentCache;
    std::optional<Layout::ConstraintsForInlineContent> m_inlineContentConstraints;
    // FIXME: This should be part of LayoutState.
    std::unique_ptr<Layout::InlineDamage> m_lineDamage;
    std::unique_ptr<InlineContent> m_inlineContent;
    BoxGeometryUpdater m_boxGeometryUpdater;
};

}
}

