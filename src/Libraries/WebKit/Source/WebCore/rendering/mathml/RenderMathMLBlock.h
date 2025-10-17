/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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

#include "MathMLElement.h"
#include "MathMLStyle.h"
#include "RenderBlock.h"
#include "RenderTable.h"

namespace WebCore {

class RenderMathMLOperator;
class MathMLPresentationElement;

class RenderMathMLBlock : public RenderBlock {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLBlock);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLBlock);
public:
    RenderMathMLBlock(Type, MathMLPresentationElement&, RenderStyle&&);
    RenderMathMLBlock(Type, Document&, RenderStyle&&);
    virtual ~RenderMathMLBlock();

    MathMLStyle& mathMLStyle() const { return m_mathMLStyle; }

    bool isChildAllowed(const RenderObject&, const RenderStyle&) const override;

    // MathML defines an "embellished operator" as roughly an <mo> that may have subscripts,
    // superscripts, underscripts, overscripts, or a denominator (as in d/dx, where "d" is some
    // differential operator). The padding, precedence, and stretchiness of the base <mo> should
    // apply to the embellished operator as a whole. unembellishedOperator() checks for being an
    // embellished operator, and omits any embellishments.
    // FIXME: We don't yet handle all the cases in the MathML spec. See
    // https://bugs.webkit.org/show_bug.cgi?id=78617.
    virtual RenderMathMLOperator* unembellishedOperator() const { return nullptr; }

    LayoutUnit baselinePosition(FontBaseline, bool firstLine, LineDirectionMode, LinePositionMode = PositionOnContainingLine) const override;

protected:
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    inline LayoutUnit ruleThicknessFallback() const;

    LayoutUnit mathAxisHeight() const;
    LayoutUnit mirrorIfNeeded(LayoutUnit horizontalOffset, LayoutUnit boxWidth = 0_lu) const;
    inline LayoutUnit mirrorIfNeeded(LayoutUnit horizontalOffset, const RenderBox& child) const;

    static inline LayoutUnit ascentForChild(const RenderBox& child);

    void layoutBlock(bool relayoutChildren, LayoutUnit pageLogicalHeight = 0_lu) override;
    void computeAndSetBlockDirectionMarginsOfChildren();
    void insertPositionedChildrenIntoContainingBlock();
    void layoutFloatingChildren();

    void shiftInFlowChildren(LayoutUnit left, LayoutUnit top);
    void adjustPreferredLogicalWidthsForBorderAndPadding();
    void adjustLayoutForBorderAndPadding();

    enum class LayoutPhase : uint8_t {
        CalculatePreferredLogicalWidth,
        Layout,
    };
    struct SizeAppliedToMathContent {
        std::optional<LayoutUnit> logicalWidth;
        std::optional<LayoutUnit> logicalHeight;
    };
    // Retrieve the specified (and supported) CSS width/height to apply to math
    // content box, if any.
    SizeAppliedToMathContent sizeAppliedToMathContent(LayoutPhase);
    // Whether math content should be centered on the inline axis if a different size is specified by the user.
    virtual bool isMathContentCentered() const { return false; }
    // Apply the specified CSS width/height to the math content box and return inline shift for further adjustments.
    LayoutUnit applySizeToMathContent(LayoutPhase, const SizeAppliedToMathContent&);

private:
    bool isRenderMathMLBlock() const final { return true; }
    ASCIILiteral renderName() const override { return "RenderMathMLBlock"_s; }
    bool canDropAnonymousBlockChild() const final { return false; }
    void layoutItems(bool relayoutChildren);

    Ref<MathMLStyle> m_mathMLStyle;
};

class RenderMathMLTable final : public RenderTable {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLTable);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLTable);
public:
    inline RenderMathMLTable(MathMLElement&, RenderStyle&&);
    virtual ~RenderMathMLTable();

    MathMLStyle& mathMLStyle() const { return m_mathMLStyle; }

private:
    ASCIILiteral renderName() const final { return "RenderMathMLTable"_s; }
    std::optional<LayoutUnit> firstLineBaseline() const final;

    Ref<MathMLStyle> m_mathMLStyle;
};

LayoutUnit toUserUnits(const MathMLElement::Length&, const RenderStyle&, const LayoutUnit& referenceValue);

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLBlock, isRenderMathMLBlock())
SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLTable, isRenderMathMLTable())

#endif // ENABLE(MATHML)
