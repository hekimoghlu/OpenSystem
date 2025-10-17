/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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

#include "FormattingGeometry.h"

namespace WebCore {
namespace Layout {

class FlexFormattingContext;
class LogicalFlexItem;

// Helper class for flex layout.
class FlexFormattingUtils {
public:
    FlexFormattingUtils(const FlexFormattingContext&);

    static bool isMainAxisParallelWithInlineAxis(const ElementBox& flexContainer);
    static bool isMainReversedToContentDirection(const ElementBox& flexContainer);
    static bool areFlexLinesReversedInCrossAxis(const ElementBox& flexContainer);

    // FIXME: These values should probably be computed by FlexFormattingContext and get passed in to FlexLayout.
    static LayoutUnit mainAxisGapValue(const ElementBox& flexContainer, LayoutUnit flexContainerContentBoxWidth);
    static LayoutUnit crossAxisGapValue(const ElementBox& flexContainer, LayoutUnit flexContainerContentBoxHeight);

    static ContentPosition logicalJustifyContentPosition(const ElementBox& flexContainer, ContentPosition);

    LayoutUnit usedMinimumSizeInMainAxis(const LogicalFlexItem&) const;
    std::optional<LayoutUnit> usedMaximumSizeInMainAxis(const LogicalFlexItem&) const;
    LayoutUnit usedMaxContentSizeInMainAxis(const LogicalFlexItem&) const;
    LayoutUnit usedSizeInCrossAxis(const LogicalFlexItem&, LayoutUnit maxAxisConstraint) const;

private:
    const FlexFormattingContext& formattingContext() const { return m_flexFormattingContext; }

private:
    const FlexFormattingContext& m_flexFormattingContext;
};

}
}

