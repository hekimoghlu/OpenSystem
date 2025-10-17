/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
#include "InlineFormattingConstraints.h"
#include "InlineFormattingUtils.h"
#include "InlineLayoutState.h"
#include "InlineQuirks.h"
#include "IntrinsicWidthHandler.h"
#include "LayoutIntegrationUtils.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace Layout {

class InlineDamage;
class InlineContentCache;
class LineBox;

struct InlineLayoutResult {
    InlineDisplay::Content displayContent;
    enum class Range : uint8_t {
        Full, // Display content represents the complete inline content -result of full layout
        FullFromDamage, // Display content represents part of the inline content starting from damaged line until the end of inline content -result of partial layout with continuous damage all the way to the end of the inline content
        PartialFromDamage // Display content represents part of the inline content starting from damaged line until damage stops -result of partial layout with damage that does not cover the entire inline content
    };
    Range range { Range::Full };
    bool didDiscardContent { false };
};

// This class implements the layout logic for inline formatting context.
// https://www.w3.org/TR/CSS22/visuren.html#inline-formatting
class InlineFormattingContext {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(InlineFormattingContext);
public:
    InlineFormattingContext(const ElementBox& formattingContextRoot, LayoutState&, BlockLayoutState& parentBlockLayoutState);

    InlineLayoutResult layout(const ConstraintsForInlineContent&, InlineDamage* = nullptr);

    std::pair<LayoutUnit, LayoutUnit> minimumMaximumContentSize(InlineDamage* = nullptr);
    LayoutUnit minimumContentSize(InlineDamage* = nullptr);
    LayoutUnit maximumContentSize(InlineDamage* = nullptr);

    const ElementBox& root() const { return m_rootBlockContainer; }
    const InlineFormattingUtils& formattingUtils() const { return m_inlineFormattingUtils; }
    const InlineQuirks& quirks() const { return m_inlineQuirks; }
    const FloatingContext& floatingContext() const { return m_floatingContext; }

    InlineLayoutState& layoutState() { return m_inlineLayoutState; }
    const InlineLayoutState& layoutState() const { return m_inlineLayoutState; }

    enum class EscapeReason {
        InkOverflowNeedsInitialContiningBlockForStrokeWidth
    };
    const BoxGeometry& geometryForBox(const Box&, std::optional<EscapeReason> = std::nullopt) const;
    BoxGeometry& geometryForBox(const Box&);

    const IntegrationUtils& integrationUtils() const { return m_integrationUtils; }

private:
    InlineLayoutResult lineLayout(AbstractLineBuilder&, const InlineItemList&, InlineItemRange, std::optional<PreviousLine>, const ConstraintsForInlineContent&, const InlineDamage* = nullptr);
    void layoutFloatContentOnly(const ConstraintsForInlineContent&);

    void collectContentIfNeeded();
    InlineRect createDisplayContentForInlineContent(const LineBox&, const LineLayoutResult&, const ConstraintsForInlineContent&, InlineDisplay::Content&, size_t& numberOfPreviousLContentfulLines);
    void updateInlineLayoutStateWithLineLayoutResult(const LineLayoutResult&, const InlineRect& lineLogicalRect, const FloatingContext&);
    void updateBoxGeometryForPlacedFloats(const LineLayoutResult::PlacedFloatList&);
    void resetBoxGeometriesForDiscardedContent(const InlineItemRange& discardedRange, const LineLayoutResult::SuspendedFloatList& suspendedFloats);
    bool createDisplayContentForLineFromCachedContent(const ConstraintsForInlineContent&, InlineLayoutResult&);
    void createDisplayContentForEmptyInlineContent(const ConstraintsForInlineContent&, InlineLayoutResult&);
    void initializeInlineLayoutState(const LayoutState&);
    void rebuildInlineItemListIfNeeded(InlineDamage*);

    InlineContentCache& inlineContentCache() { return m_inlineContentCache; }

private:
    const ElementBox& m_rootBlockContainer;
    LayoutState& m_globalLayoutState;
    const FloatingContext m_floatingContext;
    const InlineFormattingUtils m_inlineFormattingUtils;
    const InlineQuirks m_inlineQuirks;
    const IntegrationUtils m_integrationUtils;
    InlineContentCache& m_inlineContentCache;
    InlineLayoutState m_inlineLayoutState;
};

}
}
