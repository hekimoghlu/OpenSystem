/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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

#include "FloatConversion.h"
#include "FloatPoint.h"
#include "FloatSize.h"
#include "LayoutUnit.h"
#include "StylePrimitiveNumericTypes+Calculation.h"
#include "StylePrimitiveNumericTypes.h"
#include "StyleValueTypes.h"

namespace WebCore {
namespace Style {

// MARK: - Percentage

template<auto R, typename V> struct Evaluation<Percentage<R, V>> {
    constexpr double operator()(const Percentage<R, V>& percentage)
    {
        return static_cast<double>(percentage.value) / 100.0;
    }

    template<typename Reference> constexpr auto operator()(const Percentage<R, V>& percentage, Reference referenceLength) -> Reference
    {
        return static_cast<Reference>(percentage.value) / 100.0 * referenceLength;
    }
};

// MARK: - Numeric

template<NonCompositeNumeric StyleType> struct Evaluation<StyleType> {
    constexpr double operator()(const StyleType& value)
    {
        return static_cast<double>(value.value);
    }

    template<typename Reference> constexpr auto operator()(const StyleType& value, Reference) -> Reference
    {
        return static_cast<Reference>(value.value);
    }
};

// MARK: - Calculation

template<> struct Evaluation<Ref<CalculationValue>> {
    template<typename Reference> auto operator()(Ref<CalculationValue> calculation, Reference referenceLength)
    {
        return static_cast<Reference>(calculation->evaluate(referenceLength));
    }
};

template<Calc Calculation> struct Evaluation<Calculation> {
    template<typename... Rest> decltype(auto) operator()(const Calculation& calculation, Rest&&... rest)
    {
        return evaluate(calculation.protectedCalculation(), std::forward<Rest>(rest)...);
    }
};

// MARK: - SpaceSeparatedPoint

template<typename T> struct Evaluation<SpaceSeparatedPoint<T>> {
    FloatPoint operator()(const SpaceSeparatedPoint<T>& value, FloatSize referenceBox)
    {
        return {
            evaluate(value.x(), referenceBox.width()),
            evaluate(value.y(), referenceBox.height())
        };
    }
};

// MARK: - SpaceSeparatedSize

template<typename T> struct Evaluation<SpaceSeparatedSize<T>> {
    FloatSize operator()(const SpaceSeparatedSize<T>& value, FloatSize referenceBox)
    {
        return {
            evaluate(value.width(), referenceBox.width()),
            evaluate(value.height(), referenceBox.height())
        };
    }
};

// MARK: - Calculated Evaluations

// Convert to `calc(100% - value)`.
template<auto R, typename V> auto reflect(const LengthPercentage<R, V>& value) -> LengthPercentage<R, V>
{
    using Result = LengthPercentage<R, V>;
    using Dimension = typename Result::Dimension;
    using Percentage = typename Result::Percentage;
    using Calc = typename Result::Calc;

    return WTF::switchOn(value,
        [&](const Dimension& value) -> Result {
            // If `value` is 0, we can avoid the `calc` altogether.
            if (value.value == 0)
                return Percentage { 100 };

            // Turn this into a calc expression: `calc(100% - value)`.
            return Calc { Calculation::subtract(Calculation::percentage(100), copyCalculation(value)) };
        },
        [&](const Percentage& value) -> Result {
            // If `value` is a percentage, we can avoid the `calc` altogether.
            return Percentage { 100 - value.value };
        },
        [&](const Calc& value) -> Result {
            // Turn this into a calc expression: `calc(100% - value)`.
            return Calc { Calculation::subtract(Calculation::percentage(100), copyCalculation(value)) };
        }
    );
}

// Merges the two ranges, `aR` and `bR`, creating a union of their ranges.
consteval CSS::Range mergeRanges(CSS::Range aR, CSS::Range bR)
{
    return CSS::Range { std::min(aR.min, bR.min), std::max(aR.max, bR.max) };
}

// Convert to `calc(100% - (a + b))`.
//
// Returns a LengthPercentage with range, `resultR`, equal to union of the two input ranges `aR` and `bR`.
template<auto aR, auto bR, typename V> auto reflectSum(const LengthPercentage<aR, V>& a, const LengthPercentage<bR, V>& b) -> LengthPercentage<mergeRanges(aR, bR), V>
{
    constexpr auto resultR = mergeRanges(aR, bR);

    using Result = LengthPercentage<resultR, V>;
    using PercentageResult = typename Result::Percentage;
    using CalcResult = typename Result::Calc;
    using PercentageA = typename LengthPercentage<aR, V>::Percentage;
    using PercentageB = typename LengthPercentage<bR, V>::Percentage;

    bool aIsZero = a.isZero();
    bool bIsZero = b.isZero();

    // If both `a` and `b` are 0, turn this into a calc expression: `calc(100% - (0 + 0))` aka `100%`.
    if (aIsZero && bIsZero)
        return PercentageResult { 100 };

    // If just `a` is 0, we can just consider the case of `calc(100% - b)`.
    if (aIsZero) {
        return WTF::switchOn(b,
            [&](const PercentageB& b) -> Result {
                // And if `b` is a percent, we can avoid the `calc` altogether.
                return PercentageResult { 100 - b.value };
            },
            [&](const auto& b) -> Result {
                // Otherwise, turn this into a calc expression: `calc(100% - b)`.
                return CalcResult { Calculation::subtract(Calculation::percentage(100), copyCalculation(b)) };
            }
        );
    }

    // If just `b` is 0, we can just consider the case of `calc(100% - a)`.
    if (bIsZero) {
        return WTF::switchOn(a,
            [&](const PercentageA& a) -> Result {
                // And if `a` is a percent, we can avoid the `calc` altogether.
                return PercentageResult { 100 - a.value };
            },
            [&](const auto& a) -> Result {
                // Otherwise, turn this into a calc expression: `calc(100% - a)`.
                return CalcResult { Calculation::subtract(Calculation::percentage(100), copyCalculation(a)) };
            }
        );
    }

    // If both and `a` and `b` are percentages, we can avoid the `calc` altogether.
    if (WTF::holdsAlternative<PercentageA>(a) && WTF::holdsAlternative<PercentageB>(b))
        return PercentageResult { 100 - (get<PercentageA>(a).value + get<PercentageB>(b).value) };

    // Otherwise, turn this into a calc expression: `calc(100% - (a + b))`.
    return CalcResult { Calculation::subtract(Calculation::percentage(100), Calculation::add(copyCalculation(a), copyCalculation(b))) };
}

} // namespace Style
} // namespace WebCore
