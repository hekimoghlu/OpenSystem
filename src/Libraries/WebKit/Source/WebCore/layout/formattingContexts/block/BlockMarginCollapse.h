/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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

#include "MarginTypes.h"

namespace WebCore {
namespace Layout {

class BlockFormattingGeometry;
class BlockFormattingState;
class ElementBox;
class LayoutState;

// This class implements margin collapsing for block formatting context.
class BlockMarginCollapse {
public:
    BlockMarginCollapse(const LayoutState&, const BlockFormattingState&);

    UsedVerticalMargin collapsedVerticalValues(const ElementBox&, UsedVerticalMargin::NonCollapsedValues);

    LayoutUnit marginBeforeIgnoringCollapsingThrough(const ElementBox&, UsedVerticalMargin::NonCollapsedValues);

    bool marginBeforeCollapsesWithParentMarginBefore(const ElementBox&) const;
    bool marginBeforeCollapsesWithFirstInFlowChildMarginBefore(const ElementBox&) const;
    bool marginBeforeCollapsesWithParentMarginAfter(const ElementBox&) const;
    bool marginBeforeCollapsesWithPreviousSiblingMarginAfter(const ElementBox&) const;

    bool marginAfterCollapsesWithParentMarginAfter(const ElementBox&) const;
    bool marginAfterCollapsesWithLastInFlowChildMarginAfter(const ElementBox&) const;
    bool marginAfterCollapsesWithParentMarginBefore(const ElementBox&) const;
    bool marginAfterCollapsesWithNextSiblingMarginBefore(const ElementBox&) const;
    bool marginAfterCollapsesWithSiblingMarginBeforeWithClearance(const ElementBox&) const;

    UsedVerticalMargin::PositiveAndNegativePair::Values computedPositiveAndNegativeMargin(UsedVerticalMargin::PositiveAndNegativePair::Values, UsedVerticalMargin::PositiveAndNegativePair::Values) const;

    bool marginsCollapseThrough(const ElementBox&) const;

    PrecomputedMarginBefore precomputedMarginBefore(const ElementBox&, UsedVerticalMargin::NonCollapsedValues, const BlockFormattingGeometry&);

private:
    enum class MarginType { Before, After };
    UsedVerticalMargin::PositiveAndNegativePair::Values positiveNegativeValues(const ElementBox&, MarginType) const;
    UsedVerticalMargin::PositiveAndNegativePair::Values positiveNegativeMarginBefore(const ElementBox&, UsedVerticalMargin::NonCollapsedValues) const;
    UsedVerticalMargin::PositiveAndNegativePair::Values positiveNegativeMarginAfter(const ElementBox&, UsedVerticalMargin::NonCollapsedValues) const;

    UsedVerticalMargin::PositiveAndNegativePair::Values precomputedPositiveNegativeMarginBefore(const ElementBox&, UsedVerticalMargin::NonCollapsedValues, const BlockFormattingGeometry&) const;
    UsedVerticalMargin::PositiveAndNegativePair::Values precomputedPositiveNegativeValues(const ElementBox&, const BlockFormattingGeometry&) const;

    std::optional<LayoutUnit> marginValue(UsedVerticalMargin::PositiveAndNegativePair::Values) const;

    bool hasClearance(const ElementBox&) const;

    bool inQuirksMode() const { return m_inQuirksMode; }
    const LayoutState& layoutState() const { return m_layoutState; }
    const BlockFormattingState& formattingState() const { return m_blockFormattingState; }

    const LayoutState& m_layoutState;
    const BlockFormattingState& m_blockFormattingState;
    bool m_inQuirksMode { false };
};

}
}

