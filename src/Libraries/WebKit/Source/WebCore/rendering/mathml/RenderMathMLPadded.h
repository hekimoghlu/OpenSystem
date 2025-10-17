/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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

#include "MathMLPaddedElement.h"
#include "RenderMathMLRow.h"

namespace WebCore {

class RenderMathMLPadded final : public RenderMathMLRow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLPadded);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLPadded);
public:
    RenderMathMLPadded(MathMLPaddedElement&, RenderStyle&&);
    virtual ~RenderMathMLPadded();

private:
    ASCIILiteral renderName() const final { return "RenderMathMLPadded"_s; }

    void computePreferredLogicalWidths() final;
    void layoutBlock(bool relayoutChildren, LayoutUnit pageLogicalHeight = 0_lu) final;
    std::optional<LayoutUnit> firstLineBaseline() const final;

    MathMLPaddedElement& element() const { return static_cast<MathMLPaddedElement&>(nodeForNonAnonymous()); }
    LayoutUnit voffset() const;
    LayoutUnit lspace() const;
    LayoutUnit mpaddedWidth(LayoutUnit contentWidth) const;
    LayoutUnit mpaddedHeight(LayoutUnit contentHeight) const;
    LayoutUnit mpaddedDepth(LayoutUnit contentDepth) const;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLPadded, isRenderMathMLPadded())

#endif // ENABLE(MATHML)
