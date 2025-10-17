/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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

#include "RenderMathMLBlock.h"

namespace WebCore {

class MathMLRowElement;

class RenderMathMLRow : public RenderMathMLBlock {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLRow);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLRow);
public:
    RenderMathMLRow(Type, MathMLRowElement&, RenderStyle&&);
    MathMLRowElement& element() const;
    virtual ~RenderMathMLRow();

protected:
    void layoutBlock(bool relayoutChildren, LayoutUnit pageLogicalHeight = 0_lu) override;
    std::optional<LayoutUnit> firstLineBaseline() const override;

    void stretchVerticalOperatorsAndLayoutChildren();
    void getContentBoundingBox(LayoutUnit& width, LayoutUnit& ascent, LayoutUnit& descent) const;
    void layoutRowItems(LayoutUnit width, LayoutUnit ascent);
    LayoutUnit preferredLogicalWidthOfRowItems();
    void computePreferredLogicalWidths() override;

private:
    ASCIILiteral renderName() const override { return "RenderMathMLRow"_s; }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLRow, isRenderMathMLRow())

#endif // ENABLE(MATHML)
