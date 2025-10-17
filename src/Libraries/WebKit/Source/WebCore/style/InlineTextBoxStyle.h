/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 15, 2023.
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

#include "InlineIteratorInlineBox.h"
#include "InlineIteratorLineBox.h"
#include "RenderStyleConstants.h"

namespace WebCore {
    
class RenderStyle;

inline float wavyOffsetFromDecoration()
{
    return 1.f;
}

struct WavyStrokeParameters {
    // Distance between decoration's axis and Bezier curve's control points.
    // The height of the curve is based on this distance. Increases the curve's height
    // as fontSize increases to make the curve look better.
    float controlPointDistance { 0.f };

    // Increment used to form the diamond shape between start point (p1), control
    // points and end point (p2) along the axis of the decoration. The curve gets
    // wider as font size increases.
    float step { 0.f };
};
WavyStrokeParameters wavyStrokeParameters(float fontSize);

struct TextUnderlinePositionUnder {
    float textRunLogicalHeight { 0.f };
    // This offset value is the distance between the current text run's logical bottom and the lowest position of all the text runs
    // on line that belong to the same decorating box.
    float textRunOffsetFromBottomMost { 0.f };
};
GlyphOverflow visualOverflowForDecorations(const RenderStyle&);
GlyphOverflow visualOverflowForDecorations(const RenderStyle&, TextUnderlinePositionUnder);
GlyphOverflow visualOverflowForDecorations(const InlineIterator::LineBoxIterator&, const RenderText&, float textBoxLogicalTop, float textBoxLogicalBottom);
bool isAlignedForUnder(const RenderStyle& decoratingBoxStyle);

float underlineOffsetForTextBoxPainting(const InlineIterator::InlineBox&, const RenderStyle&);
float overlineOffsetForTextBoxPainting(const InlineIterator::InlineBox&, const RenderStyle&);

} // namespace WebCore
