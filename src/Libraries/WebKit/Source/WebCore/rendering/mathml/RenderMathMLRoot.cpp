/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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
#include "config.h"
#include "RenderMathMLRoot.h"

#if ENABLE(MATHML)

#include "FontCache.h"
#include "GraphicsContext.h"
#include "MathMLNames.h"
#include "MathMLRootElement.h"
#include "PaintInfo.h"
#include "RenderIterator.h"
#include "RenderMathMLBlockInlines.h"
#include "RenderMathMLMenclose.h"
#include "RenderMathMLOperator.h"
#include <wtf/TZoneMallocInlines.h>

static const UChar gRadicalCharacter = 0x221A;

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderMathMLRoot);

RenderMathMLRoot::RenderMathMLRoot(MathMLRootElement& element, RenderStyle&& style)
    : RenderMathMLRow(Type::MathMLRoot, element, WTFMove(style))
{
    m_radicalOperator.setOperator(RenderMathMLRoot::style(), gRadicalCharacter, MathOperator::Type::VerticalOperator);
    ASSERT(isRenderMathMLRoot());
}

RenderMathMLRoot::~RenderMathMLRoot() = default;

MathMLRootElement& RenderMathMLRoot::element() const
{
    return static_cast<MathMLRootElement&>(nodeForNonAnonymous());
}

RootType RenderMathMLRoot::rootType() const
{
    return element().rootType();
}

bool RenderMathMLRoot::isValid() const
{
    // Verify whether the list of children is valid:
    // <msqrt> child1 child2 ... childN </msqrt>
    // <mroot> base index </mroot>
    if (rootType() == RootType::SquareRoot)
        return true;

    ASSERT(rootType() == RootType::RootWithIndex);
    auto* child = firstInFlowChildBox();
    if (!child)
        return false;
    child = child->nextInFlowSiblingBox();
    return child && !child->nextInFlowSiblingBox();
}

RenderBox& RenderMathMLRoot::getBase() const
{
    ASSERT(isValid());
    ASSERT(rootType() == RootType::RootWithIndex);
    return *firstInFlowChildBox();
}

RenderBox& RenderMathMLRoot::getIndex() const
{
    ASSERT(isValid());
    ASSERT(rootType() == RootType::RootWithIndex);
    return *firstInFlowChildBox()->nextInFlowSiblingBox();
}

void RenderMathMLRoot::styleDidChange(StyleDifference diff, const RenderStyle* oldStyle)
{
    RenderMathMLRow::styleDidChange(diff, oldStyle);
    m_radicalOperator.reset(style());
}

RenderMathMLRoot::HorizontalParameters RenderMathMLRoot::horizontalParameters(LayoutUnit indexWidth)
{
    // Square roots do not require horizontal parameters.
    ASSERT(rootType() == RootType::RootWithIndex);
    HorizontalParameters parameters;

    // We try and read constants to draw the radical from the OpenType MATH and use fallback values otherwise.
    const Ref primaryFont = style().fontCascade().primaryFont();
    if (RefPtr mathData = primaryFont->mathData()) {
        parameters.kernBeforeDegree = mathData->getMathConstant(primaryFont, OpenTypeMathData::RadicalKernBeforeDegree);
        parameters.kernAfterDegree = mathData->getMathConstant(primaryFont, OpenTypeMathData::RadicalKernAfterDegree);
    } else {
        // RadicalKernBeforeDegree: No suggested value provided. OT Math Illuminated mentions 5/18 em, Gecko uses 0.
        // RadicalKernAfterDegree: Suggested value is -10/18 of em.
        parameters.kernBeforeDegree = 5 * style().fontCascade().size() / 18;
        parameters.kernAfterDegree = -10 * style().fontCascade().size() / 18;
    }
    // Apply clamping from https://w3c.github.io/mathml-core/#root-with-index
    parameters.kernBeforeDegree = std::max<LayoutUnit>(0, parameters.kernBeforeDegree);
    parameters.kernAfterDegree = std::max<LayoutUnit>(-indexWidth, parameters.kernAfterDegree);
    return parameters;
}

RenderMathMLRoot::VerticalParameters RenderMathMLRoot::verticalParameters()
{
    VerticalParameters parameters;
    // We try and read constants to draw the radical from the OpenType MATH and use fallback values otherwise.
    const Ref primaryFont = style().fontCascade().primaryFont();
    if (RefPtr mathData = primaryFont->mathData()) {
        parameters.ruleThickness = mathData->getMathConstant(primaryFont, OpenTypeMathData::RadicalRuleThickness);
        parameters.verticalGap = mathData->getMathConstant(primaryFont, style().mathStyle() == MathStyle::Normal ? OpenTypeMathData::RadicalDisplayStyleVerticalGap : OpenTypeMathData::RadicalVerticalGap);
        parameters.extraAscender = mathData->getMathConstant(primaryFont, OpenTypeMathData::RadicalExtraAscender);
        if (rootType() == RootType::RootWithIndex)
            parameters.degreeBottomRaisePercent = mathData->getMathConstant(primaryFont, OpenTypeMathData::RadicalDegreeBottomRaisePercent);
    } else {
        // RadicalVerticalGap: Suggested value is 5/4 default rule thickness.
        // RadicalDisplayStyleVerticalGap: Suggested value is default rule thickness + 1/4 x-height.
        // RadicalRuleThickness: Suggested value is default rule thickness.
        // RadicalExtraAscender: Suggested value is RadicalRuleThickness.
        // RadicalDegreeBottomRaisePercent: Suggested value is 60%.
        parameters.ruleThickness = ruleThicknessFallback();
        if (style().mathStyle() == MathStyle::Normal)
            parameters.verticalGap = parameters.ruleThickness + style().metricsOfPrimaryFont().xHeight().value_or(0) / 4;
        else
            parameters.verticalGap = 5 * parameters.ruleThickness / 4;

        if (rootType() == RootType::RootWithIndex) {
            parameters.extraAscender = parameters.ruleThickness;
            parameters.degreeBottomRaisePercent = 0.6f;
        }
    }
    return parameters;
}

void RenderMathMLRoot::computePreferredLogicalWidths()
{
    ASSERT(preferredLogicalWidthsDirty());

    if (!isValid()) {
        RenderMathMLRow::computePreferredLogicalWidths();
        return;
    }

    LayoutUnit preferredWidth = 0;
    if (rootType() == RootType::SquareRoot) {
        preferredWidth += m_radicalOperator.maxPreferredWidth();
        preferredWidth += preferredLogicalWidthOfRowItems();
    } else {
        ASSERT(rootType() == RootType::RootWithIndex);
        LayoutUnit indexPreferredWidth = getIndex().maxPreferredLogicalWidth() + marginIntrinsicLogicalWidthForChild(getIndex());
        auto horizontal = horizontalParameters(indexPreferredWidth);
        preferredWidth += horizontal.kernBeforeDegree;
        preferredWidth += indexPreferredWidth;
        preferredWidth += horizontal.kernAfterDegree;
        preferredWidth += m_radicalOperator.maxPreferredWidth();
        preferredWidth += getBase().maxPreferredLogicalWidth() + marginIntrinsicLogicalWidthForChild(getBase());
    }
    m_minPreferredLogicalWidth = m_maxPreferredLogicalWidth = preferredWidth;

    auto sizes = sizeAppliedToMathContent(LayoutPhase::CalculatePreferredLogicalWidth);
    applySizeToMathContent(LayoutPhase::CalculatePreferredLogicalWidth, sizes);

    adjustPreferredLogicalWidthsForBorderAndPadding();

    setPreferredLogicalWidthsDirty(false);
}

void RenderMathMLRoot::layoutBlock(bool relayoutChildren, LayoutUnit)
{
    ASSERT(needsLayout());

    insertPositionedChildrenIntoContainingBlock();

    if (!relayoutChildren && simplifiedLayout())
        return;

    m_radicalOperatorTop = 0;
    m_baseWidth = 0;

    if (!isValid()) {
        RenderMathMLRow::layoutBlock(relayoutChildren);
        return;
    }

    layoutFloatingChildren();

    // We layout the children, determine the vertical metrics of the base and set the logical width.
    // Note: Per the MathML specification, the children of <msqrt> are wrapped in an inferred <mrow>, which is the desired base.
    LayoutUnit baseAscent, baseDescent;
    recomputeLogicalWidth();
    computeAndSetBlockDirectionMarginsOfChildren();
    if (rootType() == RootType::SquareRoot) {
        stretchVerticalOperatorsAndLayoutChildren();
        getContentBoundingBox(m_baseWidth, baseAscent, baseDescent);
        layoutRowItems(m_baseWidth, baseAscent);
    } else {
        getBase().layoutIfNeeded();
        m_baseWidth = getBase().logicalWidth() + getBase().marginLogicalWidth();
        baseAscent = ascentForChild(getBase()) + getBase().marginBefore();
        baseDescent = getBase().logicalHeight() + getBase().marginLogicalHeight() - baseAscent;
        getIndex().layoutIfNeeded();
    }

    HorizontalParameters horizontal;
    auto vertical = verticalParameters();

    // Stretch the radical operator to cover the base height.
    // We can then determine the metrics of the radical operator + the base.
    m_radicalOperator.stretchTo(style(), baseAscent + baseDescent + vertical.verticalGap + vertical.ruleThickness);
    LayoutUnit radicalOperatorHeight = m_radicalOperator.ascent() + m_radicalOperator.descent();
    LayoutUnit indexBottomRaise { vertical.degreeBottomRaisePercent * radicalOperatorHeight };
    LayoutUnit radicalAscent = baseAscent + vertical.verticalGap + vertical.ruleThickness + vertical.extraAscender;
    LayoutUnit radicalDescent = std::max<LayoutUnit>(baseDescent, radicalOperatorHeight + vertical.extraAscender - radicalAscent);
    LayoutUnit descent = radicalDescent;
    LayoutUnit ascent = radicalAscent;

    // We set the logical width.
    if (rootType() == RootType::SquareRoot)
        setLogicalWidth(m_radicalOperator.width() + m_baseWidth);
    else {
        ASSERT(rootType() == RootType::RootWithIndex);
        LayoutUnit indexWidth = getIndex().logicalWidth() + getIndex().marginLogicalWidth();
        horizontal = horizontalParameters(indexWidth);
        setLogicalWidth(horizontal.kernBeforeDegree + indexWidth + horizontal.kernAfterDegree + m_radicalOperator.width() + m_baseWidth);
    }

    // For <mroot>, we update the metrics to take into account the index.
    LayoutUnit indexAscent, indexDescent;
    if (rootType() == RootType::RootWithIndex) {
        indexAscent = ascentForChild(getIndex()) + getIndex().marginBefore();
        indexDescent = getIndex().logicalHeight() + getIndex().marginLogicalHeight() - indexAscent;
        ascent = std::max<LayoutUnit>(radicalAscent, indexBottomRaise + indexDescent + indexAscent - descent);
    }

    // We set the final position of children.
    m_radicalOperatorTop = ascent - radicalAscent + vertical.extraAscender;
    LayoutUnit horizontalOffset = m_radicalOperator.width();
    if (rootType() == RootType::RootWithIndex)
        horizontalOffset += horizontal.kernBeforeDegree + getIndex().logicalWidth() + getIndex().marginLogicalWidth() + horizontal.kernAfterDegree;
    if (rootType() == RootType::SquareRoot) {
        LayoutPoint baseLocation(mirrorIfNeeded(horizontalOffset, m_baseWidth), ascent - baseAscent);
        for (auto* child = firstInFlowChildBox(); child; child = child->nextInFlowSiblingBox())
            child->setLocation(child->location() + baseLocation);
    } else {
        ASSERT(rootType() == RootType::RootWithIndex);
        LayoutPoint baseLocation(mirrorIfNeeded(horizontalOffset + getBase().marginStart(), getBase()), ascent - baseAscent + getBase().marginBefore());
        getBase().setLocation(baseLocation);
        LayoutPoint indexLocation(mirrorIfNeeded(horizontal.kernBeforeDegree + getIndex().marginStart(), getIndex()), ascent + descent - indexBottomRaise - indexDescent - indexAscent + getIndex().marginBefore());
        getIndex().setLocation(indexLocation);
    }

    setLogicalHeight(ascent + descent);

    auto sizes = sizeAppliedToMathContent(LayoutPhase::Layout);
    auto shift = applySizeToMathContent(LayoutPhase::Layout, sizes);
    shiftInFlowChildren(shift, 0);

    adjustLayoutForBorderAndPadding();

    layoutPositionedObjects(relayoutChildren);

    updateScrollInfoAfterLayout();

    clearNeedsLayout();
}

void RenderMathMLRoot::paint(PaintInfo& info, const LayoutPoint& paintOffset)
{
    RenderMathMLRow::paint(info, paintOffset);

    if (!firstChild() || info.context().paintingDisabled() || style().usedVisibility() != Visibility::Visible || !isValid())
        return;

    // We draw the radical operator.
    LayoutPoint radicalOperatorTopLeft = paintOffset + location();
    LayoutUnit horizontalOffset = borderAndPaddingStart();
    if (rootType() == RootType::RootWithIndex) {
        LayoutUnit indexWidth = getIndex().logicalWidth() + getIndex().marginLogicalWidth();
        auto horizontal = horizontalParameters(indexWidth);
        horizontalOffset += horizontal.kernBeforeDegree + indexWidth + horizontal.kernAfterDegree;
    }
    radicalOperatorTopLeft.move(mirrorIfNeeded(horizontalOffset, m_radicalOperator.width()), m_radicalOperatorTop);
    m_radicalOperator.paint(style(), info, radicalOperatorTopLeft);

    // We draw the radical line.
    LayoutUnit ruleThickness = verticalParameters().ruleThickness;
    if (!ruleThickness)
        return;
    GraphicsContextStateSaver stateSaver(info.context());

    info.context().setStrokeThickness(ruleThickness);
    info.context().setStrokeStyle(StrokeStyle::SolidStroke);
    info.context().setStrokeColor(style().visitedDependentColorWithColorFilter(CSSPropertyColor));
    LayoutPoint ruleOffsetFrom = paintOffset + location() + LayoutPoint(0_lu, m_radicalOperatorTop + ruleThickness / 2);
    LayoutPoint ruleOffsetTo = ruleOffsetFrom;
    horizontalOffset += m_radicalOperator.width();
    ruleOffsetFrom.move(mirrorIfNeeded(horizontalOffset), 0_lu);
    horizontalOffset += m_baseWidth;
    ruleOffsetTo.move(mirrorIfNeeded(horizontalOffset), 0_lu);
    info.context().drawLine(ruleOffsetFrom, ruleOffsetTo);
}

}

#endif // ENABLE(MATHML)
