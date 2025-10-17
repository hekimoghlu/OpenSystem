/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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

#include "LayoutUnit.h"
#include <optional>

namespace WebCore {
namespace Layout {

struct ComputedVerticalMargin {
    std::optional<LayoutUnit> before;
    std::optional<LayoutUnit> after;
};

struct UsedVerticalMargin {
    struct NonCollapsedValues {
        LayoutUnit before;
        LayoutUnit after;
    };
    NonCollapsedValues nonCollapsedValues;

    struct CollapsedValues {
        std::optional<LayoutUnit> before;
        std::optional<LayoutUnit> after;
        bool isCollapsedThrough { false };
    };
    CollapsedValues collapsedValues;

    // FIXME: This structure might need to change to indicate that the cached value is not necessarily the same as the box's computed margin value.
    // This only matters in case of collapse through margins when they collapse into another sibling box.
    // <div style="margin: 1px"></div><div style="margin: 10px"></div> <- the second div's before/after marings collapse through and the same time they collapse into
    // the first div. When the parent computes its before margin, it should see the second div's collapsed through margin as the value to collapse width (adjoining margin value).
    // So while the first div's before margin is not 10px, the cached value is 10px so that when we compute the parent's margin we just need to check the first
    // inflow child's cached margin values.
    struct PositiveAndNegativePair {
        struct Values {
            bool isNonZero() const { return positive.value_or(0) || negative.value_or(0); }

            std::optional<LayoutUnit> positive;
            std::optional<LayoutUnit> negative;
            bool isQuirk { false };
        };
        Values before;
        Values after;
    };
    PositiveAndNegativePair positiveAndNegativeValues;
};

static inline LayoutUnit marginBefore(const UsedVerticalMargin& usedVerticalMargin)
{
    return usedVerticalMargin.collapsedValues.before.value_or(usedVerticalMargin.nonCollapsedValues.before);
}

static inline LayoutUnit marginAfter(const UsedVerticalMargin& usedVerticalMargin)
{
    if (usedVerticalMargin.collapsedValues.isCollapsedThrough)
        return 0_lu;
    return usedVerticalMargin.collapsedValues.after.value_or(usedVerticalMargin.nonCollapsedValues.after);
}

struct ComputedHorizontalMargin {
    std::optional<LayoutUnit> start;
    std::optional<LayoutUnit> end;
};

struct UsedHorizontalMargin {
    LayoutUnit start;
    LayoutUnit end;
};

struct PrecomputedMarginBefore {
    LayoutUnit usedValue() const { return collapsedValue.value_or(nonCollapsedValue); }
    LayoutUnit nonCollapsedValue;
    std::optional<LayoutUnit> collapsedValue;
    UsedVerticalMargin::PositiveAndNegativePair::Values positiveAndNegativeMarginBefore;
};

}
}
