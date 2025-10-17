/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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

#include "CSSPropertyParserConsumer+LengthDefinitions.h"
#include "CSSPropertyParserConsumer+MetaConsumerDefinitions.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

struct LengthPercentageValidator {
    static constexpr std::optional<CSS::LengthPercentageUnit> validate(CSSUnitType unitType, CSSPropertyParserOptions options)
    {
        // NOTE: Percentages are handled explicitly by the PercentageValidator, so this only
        // needs to be concerned with the Length units.
        if (auto result = LengthValidator::validate(unitType, options))
            return static_cast<CSS::LengthPercentageUnit>(*result);
        return std::nullopt;
    }

    template<auto R, typename V> static bool isValid(CSS::LengthPercentageRaw<R, V> raw, CSSPropertyParserOptions)
    {
        // Values other than 0 and +/-âˆž are not supported for <length-percentage> numeric ranges currently.
        return isValidNonCanonicalizableDimensionValue(raw);
    }
};

template<auto R, typename V> struct ConsumerDefinition<CSS::LengthPercentage<R, V>> {
    using FunctionToken = FunctionConsumerForCalcValues<CSS::LengthPercentage<R, V>>;
    using DimensionToken = DimensionConsumer<CSS::LengthPercentage<R, V>, LengthPercentageValidator>;
    using PercentageToken = PercentageConsumer<CSS::LengthPercentage<R, V>, LengthPercentageValidator>;
    using NumberToken = NumberConsumerForUnitlessValues<CSS::LengthPercentage<R, V>, LengthPercentageValidator, CSS::LengthUnit::Px>;
};

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
