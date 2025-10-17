/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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

#if ENABLE(MATHML)

#include "MathMLFractionElement.h"
#include "RenderMathMLRow.h"

namespace WebCore {

class MathMLFractionElement;

class RenderMathMLFraction final : public RenderMathMLRow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLFraction);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLFraction);
public:
    RenderMathMLFraction(MathMLFractionElement&, RenderStyle&&);
    virtual ~RenderMathMLFraction();

    LayoutUnit defaultLineThickness() const;
    LayoutUnit lineThickness() const;
    float relativeLineThickness() const;

private:
    ASCIILiteral renderName() const final { return "RenderMathMLFraction"_s; }

    void computePreferredLogicalWidths() final;
    void layoutBlock(bool relayoutChildren, LayoutUnit pageLogicalHeight = 0_lu) final;
    std::optional<LayoutUnit> firstLineBaseline() const final;
    void paint(PaintInfo&, const LayoutPoint&) final;
    RenderMathMLOperator* unembellishedOperator() const final;
    bool isMathContentCentered() const final { return true; }

    MathMLFractionElement& element() const { return static_cast<MathMLFractionElement&>(nodeForNonAnonymous()); }

    bool isValid() const;
    RenderBox& numerator() const;
    RenderBox& denominator() const;
    LayoutUnit horizontalOffset(RenderBox&, MathMLFractionElement::FractionAlignment) const;
    struct FractionParameters {
        LayoutUnit numeratorShiftUp;
        LayoutUnit denominatorShiftDown;
    };
    FractionParameters fractionParameters() const;
    FractionParameters stackParameters() const;
    LayoutUnit fractionAscent() const;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLFraction, isRenderMathMLFraction())

#endif // ENABLE(MATHML)
