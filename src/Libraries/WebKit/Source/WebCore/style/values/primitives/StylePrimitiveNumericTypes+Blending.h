/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 4, 2024.
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

#include "AnimationUtilities.h"
#include "StylePrimitiveNumericTypes+Calculation.h"
#include "StylePrimitiveNumericTypes.h"

namespace WebCore {
namespace Style {

// MARK: Interpolation of base numeric types
// https://drafts.csswg.org/css-values/#combining-values
template<Numeric StyleType> struct Blending<StyleType> {
    constexpr auto canBlend(const StyleType&, const StyleType&) -> bool
    {
        return true;
    }

    auto blend(const StyleType& from, const StyleType& to, const BlendingContext& context) -> StyleType
    {
        if (!context.progress && context.isReplace())
            return from;

        if (context.progress == 1 && context.isReplace())
            return to;

        // FIXME: As interpolation may result in a value outside of the range allowed by the
        // primitive, we clamp the value back down to the allowed range. The spec states that
        // in some cases, an accumulated intermediate value should be allowed to be out of the
        // allowed range until after interpolation has completed, but we currently don't have
        // that concept, and the `WebCore::Length` code path did clamping in the same fashion.
        // https://drafts.csswg.org/css-values/#combining-range

        return StyleType { CSS::clampToRange<StyleType::range>(WebCore::blend(from.value, to.value, context)) };
    }
};

// MARK: Interpolation of mixed numeric types
// https://drafts.csswg.org/css-values/#combine-mixed
template<auto R, typename V> struct Blending<LengthPercentage<R, V>> {
    constexpr auto canBlend(const LengthPercentage<R, V>&, const LengthPercentage<R, V>&) -> bool
    {
        return true;
    }

    auto blend(const LengthPercentage<R, V>& from, const LengthPercentage<R, V>& to, const BlendingContext& context) -> LengthPercentage<R, V>
    {
        using Length = typename LengthPercentage<R, V>::Dimension;
        using Percentage = typename LengthPercentage<R, V>::Percentage;
        using Calc = typename LengthPercentage<R, V>::Calc;

        // Interpolation of dimension-percentage value combinations (e.g. <length-percentage>, <frequency-percentage>,
        // <angle-percentage>, <time-percentage> or equivalent notations) is defined as:
        //
        //  - equivalent to interpolation of <length> if both VA and VB are pure <length> values
        //  - equivalent to interpolation of <percentage> if both VA and VB are pure <percentage> values
        //  - equivalent to converting both values into a calc() expression representing the sum of the
        //    dimension type and a percentage (each possibly zero) and interpolating each component
        //    individually (as a <length>/<frequency>/<angle>/<time> and as a <percentage>, respectively)

        if (WTF::holdsAlternative<Calc>(from) || WTF::holdsAlternative<Calc>(to) || (from.index() != to.index())) {
            if (context.compositeOperation != CompositeOperation::Replace)
                return Calc { Calculation::add(copyCalculation(from), copyCalculation(to)) };

            // 0% to 0px -> calc(0px + 0%) to calc(0px + 0%) -> 0px
            // 0px to 0% -> calc(0px + 0%) to calc(0px + 0%) -> 0px
            if (from.isZero() && to.isZero())
                return Length { 0 };

            if (!WTF::holdsAlternative<Calc>(to) && !WTF::holdsAlternative<Percentage>(from) && (context.progress == 1 || from.isZero())) {
                if (WTF::holdsAlternative<Length>(to))
                    return WebCore::Style::blend(Length { 0 }, get<Length>(to), context);
                return WebCore::Style::blend(Percentage { 0 }, get<Percentage>(to), context);
            }

            if (!WTF::holdsAlternative<Calc>(from) && !WTF::holdsAlternative<Percentage>(to) && (!context.progress || to.isZero())) {
                if (WTF::holdsAlternative<Length>(from))
                    return WebCore::Style::blend(get<Length>(from), Length { 0 }, context);
                return WebCore::Style::blend(get<Percentage>(from), Percentage { 0 }, context);
            }

            return Calc { Calculation::blend(copyCalculation(from), copyCalculation(to), context.progress) };
        }

        if (!context.progress && context.isReplace())
            return from;

        if (context.progress == 1 && context.isReplace())
            return to;

        if (WTF::holdsAlternative<Length>(to))
            return WebCore::Style::blend(get<Length>(from), get<Length>(to), context);
        return WebCore::Style::blend(get<Percentage>(from), get<Percentage>(to), context);
    }
};

// `NumberOrPercentageResolvedToNumber<nR, pR, V>` forwards to `Number<nR, V>`.
template<auto nR, auto pR, typename V> struct Blending<NumberOrPercentageResolvedToNumber<nR, pR, V>> {
    auto canBlend(const NumberOrPercentageResolvedToNumber<nR, pR, V>& a, const NumberOrPercentageResolvedToNumber<nR, pR, V>& b) -> bool
    {
        return Style::canBlend(a.value, b.value);
    }
    auto blend(const NumberOrPercentageResolvedToNumber<nR, pR, V>& a, const NumberOrPercentageResolvedToNumber<nR, pR, V>& b, const BlendingContext& context) -> NumberOrPercentageResolvedToNumber<nR, pR, V>
    {
        return Style::blend(a.value, b.value, context);
    }
};

} // namespace Style
} // namespace WebCore
