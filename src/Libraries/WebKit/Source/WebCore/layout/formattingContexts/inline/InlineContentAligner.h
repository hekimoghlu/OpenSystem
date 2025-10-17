/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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
#include "InlineDisplayLine.h"
#include "LayoutUnits.h"

namespace WebCore {
namespace Layout {

class InlineContentAligner {
public:
    static InlineLayoutUnit applyTextAlignJustify(Line::RunList&, InlineLayoutUnit spaceToDistribute, size_t hangingTrailingWhitespaceLength);

    static InlineLayoutUnit applyRubyAlign(RubyAlign, Line::RunList&, WTF::Range<size_t>, InlineLayoutUnit spaceToDistribute);

    enum class AdjustContentOnlyInsideRubyBase : bool { No, Yes };
    static void applyRubyBaseAlignmentOffset(InlineDisplay::Boxes&, const UncheckedKeyHashMap<const Box*, InlineLayoutUnit>& alignmentOffsetList, AdjustContentOnlyInsideRubyBase, InlineFormattingContext&);
    static void applyRubyAnnotationAlignmentOffset(InlineDisplay::Boxes&, InlineLayoutUnit alignmentOffset, InlineFormattingContext&);

private:
    static InlineLayoutUnit applyExpansionOnRange(Line::RunList&, WTF::Range<size_t>, const ExpansionInfo&, InlineLayoutUnit spaceToDistribute);
};

}
}

