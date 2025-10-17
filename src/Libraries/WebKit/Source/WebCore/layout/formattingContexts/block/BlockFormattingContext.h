/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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

#include "BlockFormattingGeometry.h"
#include "BlockFormattingQuirks.h"
#include "BlockFormattingState.h"
#include "FormattingContext.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class LayoutUnit;

namespace Layout {

class Box;
class BlockMarginCollapse;
class FloatingContext;

// This class implements the layout logic for block formatting contexts.
// https://www.w3.org/TR/CSS22/visuren.html#block-formatting
class BlockFormattingContext : public FormattingContext {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(BlockFormattingContext);
public:
    BlockFormattingContext(const ElementBox& formattingContextRoot, BlockFormattingState&);

    void layoutInFlowContent(const ConstraintsForInFlowContent&) override;
    void layoutOutOfFlowContent(const ConstraintsForOutOfFlowContent&);
    LayoutUnit usedContentHeight() const override;

    const BlockFormattingState& formattingState() const { return m_blockFormattingState; }
    const BlockFormattingGeometry& formattingGeometry() const { return m_blockFormattingGeometry; }
    const BlockFormattingQuirks& formattingQuirks() const { return m_blockFormattingQuirks; }

protected:
    struct ConstraintsPair {
        ConstraintsForInFlowContent formattingContextRoot;
        ConstraintsForInFlowContent containingBlock;
    };
    void placeInFlowPositionedChildren(const ElementBox&, const HorizontalConstraints&);

    void computeWidthAndMargin(const FloatingContext&, const ElementBox&, const ConstraintsPair&);
    void computeHeightAndMargin(const ElementBox&, const ConstraintsForInFlowContent&);

    void computeStaticHorizontalPosition(const ElementBox&, const HorizontalConstraints&);
    void computeStaticVerticalPosition(const ElementBox&, LayoutUnit containingBlockContentBoxTop);
    void computePositionToAvoidFloats(const FloatingContext&, const ElementBox&, const ConstraintsPair&);
    void computeVerticalPositionForFloatClear(const FloatingContext&, const ElementBox&);

    void precomputeVerticalPositionForBoxAndAncestors(const ElementBox&, const ConstraintsPair&);

    IntrinsicWidthConstraints computedIntrinsicWidthConstraints() override;

    LayoutUnit verticalPositionWithMargin(const ElementBox&, const UsedVerticalMargin&, LayoutUnit containingBlockContentBoxTop) const;

    std::optional<LayoutUnit> usedAvailableWidthForFloatAvoider(const FloatingContext&, const ElementBox&, const ConstraintsPair&);
    void updateMarginAfterForPreviousSibling(const ElementBox&);

    void collectOutOfFlowDescendantsIfNeeded();
    void computeOutOfFlowVerticalGeometry(const Box&, const ConstraintsForOutOfFlowContent&);
    void computeOutOfFlowHorizontalGeometry(const Box&, const ConstraintsForOutOfFlowContent&);
    void computeBorderAndPadding(const Box&, const HorizontalConstraints&);

    BlockFormattingState& formattingState() { return m_blockFormattingState; }
    BlockMarginCollapse marginCollapse() const;

#if ASSERT_ENABLED
    void setPrecomputedMarginBefore(const ElementBox& layoutBox, const PrecomputedMarginBefore& precomputedMarginBefore) { m_precomputedMarginBeforeList.set(&layoutBox, precomputedMarginBefore); }
    PrecomputedMarginBefore precomputedMarginBefore(const ElementBox& layoutBox) const { return m_precomputedMarginBeforeList.get(&layoutBox); }
    bool hasPrecomputedMarginBefore(const ElementBox& layoutBox) const { return m_precomputedMarginBeforeList.contains(&layoutBox); }
#endif

private:
#if ASSERT_ENABLED
    UncheckedKeyHashMap<const ElementBox*, PrecomputedMarginBefore> m_precomputedMarginBeforeList;
#endif
    BlockFormattingState& m_blockFormattingState;
    const BlockFormattingGeometry m_blockFormattingGeometry;
    const BlockFormattingQuirks m_blockFormattingQuirks;
};

}
}

SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_CONTEXT(BlockFormattingContext, isBlockFormattingContext())

