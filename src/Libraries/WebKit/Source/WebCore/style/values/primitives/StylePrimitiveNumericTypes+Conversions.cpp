/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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
#include "StylePrimitiveNumericTypes+Conversions.h"

#include "RenderStyleInlines.h"
#include "StyleLengthResolution.h"

namespace WebCore {
namespace Style {

// MARK: Length Canonicalization

double canonicalizeLength(double value, CSS::LengthUnit unit, NoConversionDataRequiredToken)
{
    return computeNonCalcLengthDouble(value, unit, { });
}

double canonicalizeLength(double value, CSS::LengthUnit unit, const CSSToLengthConversionData& conversionData)
{
    return computeNonCalcLengthDouble(value, unit, conversionData);
}

float clampLengthToAllowedLimits(double value)
{
    return clampTo<float>(narrowPrecisionToFloat(value), minValueForCssLength, maxValueForCssLength);
}

float canonicalizeAndClampLength(double value, CSS::LengthUnit unit, NoConversionDataRequiredToken token)
{
    return clampLengthToAllowedLimits(canonicalizeLength(value, unit, token));
}

float canonicalizeAndClampLength(double value, CSS::LengthUnit unit, const CSSToLengthConversionData& conversionData)
{
    return clampLengthToAllowedLimits(canonicalizeLength(value, unit, conversionData));
}

// MARK: ToCSS utilities

float adjustForZoom(float value, const RenderStyle& style)
{
    return adjustFloatForAbsoluteZoom(value, style);
}

} // namespace Style
} // namespace WebCore
