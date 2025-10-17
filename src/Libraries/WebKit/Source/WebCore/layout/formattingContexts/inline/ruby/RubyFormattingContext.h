/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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

#include "InlineContentBreaker.h"
#include "InlineDisplayContent.h"
#include "InlineLevelBox.h"
#include <wtf/Range.h>

namespace WebCore {
namespace Layout {

class InlineFormattingContext;
class Line;
class LineBox;
class Rect;

class RubyFormattingContext {
public:
    // Line building
    static InlineLayoutUnit annotationBoxLogicalWidth(const Box& rubyBaseLayoutBox, InlineFormattingContext&);
    static InlineLayoutUnit baseEndAdditionalLogicalWidth(const Box& rubyBaseLayoutBox, const Line::RunList&, const InlineContentBreaker::ContinuousContent::RunList&, InlineFormattingContext&);
    static UncheckedKeyHashMap<const Box*, InlineLayoutUnit> applyRubyAlign(Line&, InlineFormattingContext&);
    static InlineLayoutUnit applyRubyAlignOnAnnotationBox(Line&, InlineLayoutUnit spaceToDistribute, InlineFormattingContext&);

    // Line box building
    static void applyAnnotationContributionToLayoutBounds(LineBox&, const InlineFormattingContext&);

    // Display content building
    static InlineLayoutUnit baseEndAdditionalLogicalWidth(const Box& rubyBaseLayoutBox, const InlineDisplay::Box& baseDisplayBox, InlineLayoutUnit baseContentWidth, InlineFormattingContext&);
    static InlineLayoutPoint placeAnnotationBox(const Box& rubyBaseLayoutBox, const Rect& rubyBaseMarginBoxLogicalRect, InlineFormattingContext&);
    static InlineLayoutSize sizeAnnotationBox(const Box& rubyBaseLayoutBox, const Rect& rubyBaseMarginBoxLogicalRect, InlineFormattingContext&);

    static void applyRubyOverhang(InlineFormattingContext& parentFormattingContext, InlineLayoutUnit lineLogicalHeight, InlineDisplay::Boxes&, const Vector<WTF::Range<size_t>>& interlinearRubyColumnRangeList);

    enum class RubyBasesMayNeedResizing : bool { No, Yes };
    static void applyAlignmentOffsetList(InlineDisplay::Boxes&, const UncheckedKeyHashMap<const Box*, InlineLayoutUnit>& alignmentOffsetList, RubyBasesMayNeedResizing, InlineFormattingContext&);
    static void applyAnnotationAlignmentOffset(InlineDisplay::Boxes&, InlineLayoutUnit alignmentOffset, InlineFormattingContext&);

    // Miscellaneous helpers
    static bool hasInterlinearAnnotation(const Box& rubyBaseLayoutBox);
    static bool hasInterCharacterAnnotation(const Box& rubyBaseLayoutBox);

private:
    using MaximumLayoutBoundsStretchMap = UncheckedKeyHashMap<const InlineLevelBox*, InlineLevelBox::AscentAndDescent>;
    static void adjustLayoutBoundsAndStretchAncestorRubyBase(LineBox&, InlineLevelBox& rubyBaseInlineBox, MaximumLayoutBoundsStretchMap&, const InlineFormattingContext&);

    static size_t applyRubyAlignOnBaseContent(size_t rubyBaseStart, Line&, UncheckedKeyHashMap<const Box*, InlineLayoutUnit>& alignmentOffsetList, InlineFormattingContext&);
    static InlineLayoutUnit overhangForAnnotationBefore(const Box& rubyBaseLayoutBox, size_t rubyBaseStart, const InlineDisplay::Boxes&, InlineLayoutUnit lineLogicalHeight, InlineFormattingContext&);
    static InlineLayoutUnit overhangForAnnotationAfter(const Box& rubyBaseLayoutBox, WTF::Range<size_t> rubyBaseRange, const InlineDisplay::Boxes&, InlineLayoutUnit lineLogicalHeight, InlineFormattingContext&);
};

} // namespace Layout
} // namespace WebCore
