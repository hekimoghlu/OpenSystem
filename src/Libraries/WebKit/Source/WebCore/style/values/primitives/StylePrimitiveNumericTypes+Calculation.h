/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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

#include "CalculationValue.h"
#include "StylePrimitiveNumericTypes.h"
#include <wtf/Forward.h>

namespace WebCore {
namespace Style {

// MARK: - Conversion to `Calculation::Child`.

inline Calculation::Child copyCalculation(Ref<CalculationValue> value)
{
    return value->copyRoot();
}

inline Calculation::Child copyCalculation(Calc auto const& value)
{
    return value.protectedCalculation()->copyRoot();
}

template<auto R, typename V> Calculation::Child copyCalculation(const Number<R, V>& value)
{
    return Calculation::number(value.value);
}

template<auto R, typename V> Calculation::Child copyCalculation(const Percentage<R, V>& value)
{
    return Calculation::percentage(value.value);
}

inline Calculation::Child copyCalculation(Numeric auto const& value)
{
    return Calculation::dimension(value.value);
}

inline Calculation::Child copyCalculation(DimensionPercentageNumeric auto const& value)
{
    return WTF::switchOn(value, [](const auto& value) { return copyCalculation(value); });
}

} // namespace Style
} // namespace WebCore
