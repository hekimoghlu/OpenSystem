/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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
#include "TextSizeAdjustment.h"

#if ENABLE(TEXT_AUTOSIZING)

#include "RenderStyle.h"
#include "RenderStyleInlines.h"

namespace WebCore {

bool AutosizeStatus::probablyContainsASmallFixedNumberOfLines(const RenderStyle& style)
{
    auto& lineHeightAsLength = style.specifiedLineHeight();
    if (!lineHeightAsLength.isFixed() && !lineHeightAsLength.isPercent())
        return false;

    auto& maxHeight = style.maxHeight();
    std::optional<Length> heightOrMaxHeightAsLength;
    if (maxHeight.isFixed())
        heightOrMaxHeightAsLength = style.maxHeight();
    else if (style.height().isFixed() && (!maxHeight.isSpecified() || maxHeight.isUndefined()))
        heightOrMaxHeightAsLength = style.height();

    if (!heightOrMaxHeightAsLength)
        return false;

    float heightOrMaxHeight = heightOrMaxHeightAsLength->value();
    if (heightOrMaxHeight <= 0)
        return false;

    float approximateLineHeight = lineHeightAsLength.isPercent() ? lineHeightAsLength.percent() * style.specifiedFontSize() / 100 : lineHeightAsLength.value();
    if (approximateLineHeight <= 0)
        return false;

    float approximateNumberOfLines = heightOrMaxHeight / approximateLineHeight;
    auto& lineClamp = style.lineClamp();
    if (!lineClamp.isNone() && !lineClamp.isPercentage()) {
        int lineClampValue = lineClamp.value();
        return lineClampValue && std::floor(approximateNumberOfLines) == lineClampValue;
    }

    const int maximumNumberOfLines = 5;
    const float thresholdForConsideringAnApproximateNumberOfLinesToBeCloseToAnInteger = 0.01;
    return approximateNumberOfLines <= maximumNumberOfLines + thresholdForConsideringAnApproximateNumberOfLinesToBeCloseToAnInteger
        && approximateNumberOfLines - std::floor(approximateNumberOfLines) <= thresholdForConsideringAnApproximateNumberOfLinesToBeCloseToAnInteger;
}

auto AutosizeStatus::computeStatus(const RenderStyle& style) -> AutosizeStatus
{
    auto result = style.autosizeStatus().fields();

    auto shouldAvoidAutosizingEntireSubtree = [&] {
        if (style.display() == DisplayType::None)
            return true;

        const float maximumDifferenceBetweenFixedLineHeightAndFontSize = 5;
        auto& lineHeight = style.specifiedLineHeight();
        if (lineHeight.isFixed() && lineHeight.value() - style.specifiedFontSize() > maximumDifferenceBetweenFixedLineHeightAndFontSize)
            return false;

        if (style.whiteSpaceCollapse() == WhiteSpaceCollapse::Collapse && style.textWrapMode() == TextWrapMode::NoWrap)
            return false;

        return probablyContainsASmallFixedNumberOfLines(style);
    };

    if (shouldAvoidAutosizingEntireSubtree())
        result.add(Fields::AvoidSubtree);

    if (style.height().isFixed())
        result.add(Fields::FixedHeight);

    if (style.width().isFixed())
        result.add(Fields::FixedWidth);

    if (style.overflowX() == Overflow::Hidden)
        result.add(Fields::OverflowXHidden);

    if (style.isFloating())
        result.add(Fields::Floating);

    return AutosizeStatus(result);
}

void AutosizeStatus::updateStatus(RenderStyle& style)
{
    style.setAutosizeStatus(AutosizeStatus(computeStatus(style)));
}

float AutosizeStatus::idempotentTextSize(float specifiedSize, float pageScale)
{
    if (pageScale >= 1)
        return specifiedSize;

    // This describes a piecewise curve when the page scale is 2/3.
    static constexpr std::array points = {
        FloatPoint { 0.0f, 0.0f },
        FloatPoint { 6.0f, 9.0f },
        FloatPoint { 14.0f, 17.0f }
    };

    // When the page scale is 1, the curve should be the identity.
    // Linearly interpolate between the curve above and identity based on the page scale.
    // Beware that depending on the specific values picked in the curve, this interpolation might change the shape of the curve for very small pageScales.
    pageScale = std::min(std::max(pageScale, 0.5f), 1.0f);
    auto scalePoint = [&](FloatPoint point) {
        float fraction = 3.0f - 3.0f * pageScale;
        point.setY(point.x() + (point.y() - point.x()) * fraction);
        return point;
    };

    if (specifiedSize <= 0)
        return 0;

    float result = scalePoint(points.back()).y();
    for (size_t i = 1; i < points.size(); ++i) {
        if (points[i].x() < specifiedSize)
            continue;
        auto leftPoint = scalePoint(points[i - 1]);
        auto rightPoint = scalePoint(points[i]);
        float fraction = (specifiedSize - leftPoint.x()) / (rightPoint.x() - leftPoint.x());
        result = leftPoint.y() + fraction * (rightPoint.y() - leftPoint.y());
        break;
    }

    return std::max(std::round(result), specifiedSize);
}

}

#endif
