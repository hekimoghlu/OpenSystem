/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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

#include "CSSValueKeywords.h"
#include "FontSelectionAlgorithm.h"

namespace WebCore {

inline std::optional<FontSelectionValue> fontWeightValue(CSSValueID value)
{
    switch (value) {
    case CSSValueNormal:
        return normalWeightValue();
    case CSSValueBold:
    case CSSValueBolder:
        return boldWeightValue();
    case CSSValueLighter:
        return lightWeightValue();
    default:
        return std::nullopt;
    }
}

inline std::optional<CSSValueID> fontWidthKeyword(FontSelectionValue width)
{
    if (width == ultraCondensedWidthValue())
        return CSSValueUltraCondensed;
    if (width == extraCondensedWidthValue())
        return CSSValueExtraCondensed;
    if (width == condensedWidthValue())
        return CSSValueCondensed;
    if (width == semiCondensedWidthValue())
        return CSSValueSemiCondensed;
    if (width == normalWidthValue())
        return CSSValueNormal;
    if (width == semiExpandedWidthValue())
        return CSSValueSemiExpanded;
    if (width == expandedWidthValue())
        return CSSValueExpanded;
    if (width == extraExpandedWidthValue())
        return CSSValueExtraExpanded;
    if (width == ultraExpandedWidthValue())
        return CSSValueUltraExpanded;
    return std::nullopt;
}

inline std::optional<FontSelectionValue> fontWidthValue(CSSValueID value)
{
    switch (value) {
    case CSSValueUltraCondensed:
        return ultraCondensedWidthValue();
    case CSSValueExtraCondensed:
        return extraCondensedWidthValue();
    case CSSValueCondensed:
        return condensedWidthValue();
    case CSSValueSemiCondensed:
        return semiCondensedWidthValue();
    case CSSValueNormal:
        return normalWidthValue();
    case CSSValueSemiExpanded:
        return semiExpandedWidthValue();
    case CSSValueExpanded:
        return expandedWidthValue();
    case CSSValueExtraExpanded:
        return extraExpandedWidthValue();
    case CSSValueUltraExpanded:
        return ultraExpandedWidthValue();
    default:
        return std::nullopt;
    }
}

inline std::optional<CSSValueID> fontStyleKeyword(std::optional<FontSelectionValue> style, FontStyleAxis axis)
{
    if (!style)
        return CSSValueNormal;
    if (style.value() == italicValue())
        return axis == FontStyleAxis::ital ? CSSValueItalic : CSSValueOblique;
    return std::nullopt;
}

inline FontSelectionValue normalizedFontItalicValue(float inputValue)
{
    return FontSelectionValue { std::clamp(inputValue, -90.0f, 90.0f) };
}

}
