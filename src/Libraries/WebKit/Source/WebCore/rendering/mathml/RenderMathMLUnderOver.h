/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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

#include "RenderMathMLScripts.h"

namespace WebCore {

class MathMLUnderOverElement;

class RenderMathMLUnderOver final : public RenderMathMLScripts {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLUnderOver);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLUnderOver);
public:
    RenderMathMLUnderOver(MathMLUnderOverElement&, RenderStyle&&);
    virtual ~RenderMathMLUnderOver();

private:
    bool isRenderMathMLScripts() const final { return false; }
    bool isMathContentCentered() const final { return !shouldMoveLimits(); }
    ASCIILiteral renderName() const final { return "RenderMathMLUnderOver"_s; }
    MathMLUnderOverElement& element() const;

    void computePreferredLogicalWidths() final;
    void layoutBlock(bool relayoutChildren, LayoutUnit pageLogicalHeight = 0_lu) final;

    void stretchHorizontalOperatorsAndLayoutChildren();
    bool isValid() const;
    bool shouldMoveLimits() const;
    RenderBox& base() const;
    RenderBox& under() const;
    RenderBox& over() const;
    LayoutUnit horizontalOffset(const RenderBox&) const;
    bool hasAccent(bool accentUnder = false) const;
    bool hasAccentUnder() const { return hasAccent(true); };
    struct VerticalParameters {
        bool useUnderOverBarFallBack;
        LayoutUnit underGapMin;
        LayoutUnit overGapMin;
        LayoutUnit underShiftMin;
        LayoutUnit overShiftMin;
        LayoutUnit underExtraDescender;
        LayoutUnit overExtraAscender;
        LayoutUnit accentBaseHeight;
    };
    VerticalParameters verticalParameters() const;
};

}

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLUnderOver, isRenderMathMLUnderOver())

#endif // ENABLE(MATHML)
