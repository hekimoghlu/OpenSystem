/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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

#include "MathOperator.h"
#include "RenderMathMLRow.h"

namespace WebCore {

class MathMLRootElement;

enum class RootType { SquareRoot, RootWithIndex };

// Render base^(1/index), or sqrt(base) using radical notation.
class RenderMathMLRoot final : public RenderMathMLRow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLRoot);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLRoot);
public:
    RenderMathMLRoot(MathMLRootElement&, RenderStyle&&);
    virtual ~RenderMathMLRoot();

    void updateStyle();

private:
    bool isValid() const;
    RenderBox& getBase() const;
    RenderBox& getIndex() const;
    ASCIILiteral renderName() const final { return "RenderMathMLRoot"_s; }
    MathMLRootElement& element() const;
    RootType rootType() const;

    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) final;

    void computePreferredLogicalWidths() final;
    void layoutBlock(bool relayoutChildren, LayoutUnit pageLogicalHeight = 0_lu) final;
    void paint(PaintInfo&, const LayoutPoint&) final;

    struct HorizontalParameters {
        LayoutUnit kernBeforeDegree;
        LayoutUnit kernAfterDegree;
    };
    HorizontalParameters horizontalParameters(LayoutUnit indexWidth);
    struct VerticalParameters {
        LayoutUnit verticalGap;
        LayoutUnit ruleThickness;
        LayoutUnit extraAscender;
        float degreeBottomRaisePercent;
    };
    VerticalParameters verticalParameters();

    MathOperator m_radicalOperator;
    LayoutUnit m_radicalOperatorTop;
    LayoutUnit m_baseWidth;

    bool isRenderMathMLSquareRoot() const final { return rootType() == RootType::SquareRoot; }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLRoot, isRenderMathMLRoot())

#endif // ENABLE(MATHML)
