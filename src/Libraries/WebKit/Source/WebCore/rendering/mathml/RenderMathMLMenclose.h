/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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

#include "MathMLMencloseElement.h"
#include "RenderMathMLRow.h"

namespace WebCore {

class RenderMathMLMenclose final : public RenderMathMLRow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLMenclose);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLMenclose);
public:
    RenderMathMLMenclose(MathMLMencloseElement&, RenderStyle&&);
    virtual ~RenderMathMLMenclose();

private:
    ASCIILiteral renderName() const final { return "RenderMathMLMenclose"_s; }
    void computePreferredLogicalWidths() final;
    void layoutBlock(bool relayoutChildren, LayoutUnit pageLogicalHeight = 0_lu) final;
    void paint(PaintInfo&, const LayoutPoint&) final;

    LayoutUnit ruleThickness() const;
    bool hasNotation(MathMLMencloseElement::MencloseNotationFlag notationFlag) const { return downcast<MathMLMencloseElement>(element()).hasNotation(notationFlag); }

    struct SpaceAroundContent {
        LayoutUnit left;
        LayoutUnit right;
        LayoutUnit top;
        LayoutUnit bottom;
    };
    SpaceAroundContent spaceAroundContent(LayoutUnit contentWidth, LayoutUnit contentHeight) const;

    LayoutRect m_contentRect;
};

}

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLMenclose, isRenderMathMLMenclose())

#endif // ENABLE(MATHML)
