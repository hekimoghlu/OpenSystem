/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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

#include "MathMLOperatorDictionary.h"
#include "MathOperator.h"
#include "RenderMathMLToken.h"

namespace WebCore {

class MathMLOperatorElement;

class RenderMathMLOperator : public RenderMathMLToken {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLOperator);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLOperator);
public:
    RenderMathMLOperator(Type, MathMLOperatorElement&, RenderStyle&&);
    RenderMathMLOperator(Type, Document&, RenderStyle&&);
    virtual ~RenderMathMLOperator();

    MathMLOperatorElement& element() const;

    void stretchTo(LayoutUnit heightAboveBaseline, LayoutUnit depthBelowBaseline);
    void stretchTo(LayoutUnit width);
    LayoutUnit stretchSize() const { return isVertical() ? m_stretchHeightAboveBaseline + m_stretchDepthBelowBaseline : m_stretchWidth; }
    void resetStretchSize();
    void setStretchWidthLocked(bool stretchWidthLocked) { m_isStretchWidthLocked = stretchWidthLocked; }
    bool isStretchWidthLocked() const { return m_isStretchWidthLocked; }

    virtual bool hasOperatorFlag(MathMLOperatorDictionary::Flag) const;
    bool isLargeOperatorInDisplayStyle() const { return !hasOperatorFlag(MathMLOperatorDictionary::Stretchy) && hasOperatorFlag(MathMLOperatorDictionary::LargeOp) && style().mathStyle() == MathStyle::Normal; }
    bool shouldMoveLimits() const { return hasOperatorFlag(MathMLOperatorDictionary::MovableLimits); }
    virtual bool isVertical() const;
    LayoutUnit italicCorrection() const { return m_mathOperator.italicCorrection(); }

    void updateTokenContent() final;
    void updateFromElement() final;
    virtual char32_t textContent() const;
    bool isStretchy() const { return textContent() && hasOperatorFlag(MathMLOperatorDictionary::Stretchy); }

protected:
    virtual void updateMathOperator();
    virtual LayoutUnit leadingSpace() const;
    virtual LayoutUnit trailingSpace() const;
    virtual LayoutUnit minSize() const;
    virtual LayoutUnit maxSize() const;
    virtual bool useMathOperator() const;

private:
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) final;
    void computePreferredLogicalWidths() final;
    void layoutBlock(bool relayoutChildren, LayoutUnit pageLogicalHeight = 0_lu) final;
    void paint(PaintInfo&, const LayoutPoint&) final;

    ASCIILiteral renderName() const final { return isAnonymous() ? "RenderMathMLOperator (anonymous)"_s : "RenderMathMLOperator"_s; }
    void paintChildren(PaintInfo& forSelf, const LayoutPoint&, PaintInfo& forChild, bool usePrintRect) final;
    bool isRenderMathMLOperator() const final { return true; }
    bool isInvisibleOperator() const;

    std::optional<LayoutUnit> firstLineBaseline() const final;
    RenderMathMLOperator* unembellishedOperator() const final { return const_cast<RenderMathMLOperator*>(this); }

    LayoutUnit verticalStretchedOperatorShift() const;

    LayoutUnit m_stretchHeightAboveBaseline { 0 };
    LayoutUnit m_stretchDepthBelowBaseline { 0 };
    LayoutUnit m_stretchWidth;
    bool m_isStretchWidthLocked { false };

    MathOperator m_mathOperator;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLOperator, isRenderMathMLOperator())

#endif // ENABLE(MATHML)
