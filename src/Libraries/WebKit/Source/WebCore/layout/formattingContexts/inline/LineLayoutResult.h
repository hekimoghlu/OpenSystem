/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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

#include "InlineLine.h"
#include "InlineLineTypes.h"
#include "LayoutUnits.h"
#include "PlacedFloats.h"

namespace WebCore {
namespace Layout {

struct LineLayoutResult {
    using PlacedFloatList = PlacedFloats::List;
    using SuspendedFloatList = Vector<const Box*>;

    InlineItemRange inlineItemRange;
    Line::RunList inlineContent;

    struct FloatContent {
        PlacedFloatList placedFloats;
        SuspendedFloatList suspendedFloats;
        OptionSet<UsedFloat> hasIntrusiveFloat { };
    };
    FloatContent floatContent { };

    struct ContentGeometry {
        InlineLayoutUnit logicalLeft { 0.f };
        InlineLayoutUnit logicalWidth { 0.f };
        InlineLayoutUnit logicalRightIncludingNegativeMargin { 0.f }; // Note that with negative horizontal margin value, contentLogicalLeft + contentLogicalWidth is not necessarily contentLogicalRight.
        std::optional<InlineLayoutUnit> trailingOverflowingContentWidth { };
    };
    ContentGeometry contentGeometry { };

    struct LineGeometry {
        InlineLayoutPoint logicalTopLeft;
        InlineLayoutUnit logicalWidth { 0.f };
        InlineLayoutUnit initialLogicalLeftIncludingIntrusiveFloats { 0.f };
        std::optional<InlineLayoutUnit> initialLetterClearGap { };
    };
    LineGeometry lineGeometry { };

    struct HangingContent {
        bool shouldContributeToScrollableOverflow { false };
        InlineLayoutUnit logicalWidth { 0.f };
        InlineLayoutUnit hangablePunctuationStartWidth { 0.f };
    };
    HangingContent hangingContent { };

    struct Directionality {
        Vector<int32_t> visualOrderList;
        TextDirection inlineBaseDirection { TextDirection::LTR };
    };
    Directionality directionality { };

    struct IsFirstLast {
        enum class FirstFormattedLine : uint8_t {
            No,
            WithinIFC,
            WithinBFC
        };
        FirstFormattedLine isFirstFormattedLine { FirstFormattedLine::WithinIFC };
        bool isLastLineWithInlineContent { true };
    };
    IsFirstLast isFirstLast { };

    struct Ruby {
        UncheckedKeyHashMap<const Box*, InlineLayoutUnit> baseAlignmentOffsetList { };
        InlineLayoutUnit annotationAlignmentOffset { 0.f };
    };
    Ruby ruby { };

    // Misc
    bool endsWithHyphen { false };
    size_t nonSpanningInlineLevelBoxCount { 0 };
    InlineLayoutUnit trimmedTrailingWhitespaceWidth { 0.f }; // only used for line-break: after-white-space currently
    InlineLayoutUnit firstLineStartTrim { 0.f }; // This is how much text-box-trim: start adjusts the first line box. We only need it to adjust the initial letter float position (which will not be needed once we drop the float behavior)
    std::optional<InlineLayoutUnit> hintForNextLineTopToAvoidIntrusiveFloat { }; // This is only used for cases when intrusive floats prevent any content placement at current vertical position.
};

}
}
