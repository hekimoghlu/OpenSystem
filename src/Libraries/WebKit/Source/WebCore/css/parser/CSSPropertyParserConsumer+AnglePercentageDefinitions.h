/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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

#include "CSSPropertyParserConsumer+AngleDefinitions.h"
#include "CSSPropertyParserConsumer+MetaConsumerDefinitions.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

struct AnglePercentageValidator {
    static constexpr std::optional<CSS::AnglePercentageUnit> validate(CSSUnitType unitType, CSSPropertyParserOptions options)
    {
        // NOTE: Percentages are handled explicitly by the PercentageValidator, so this only
        // needs to be concerned with the Angle units.
        if (auto result = AngleValidator::validate(unitType, options))
            return static_cast<CSS::AnglePercentageUnit>(*result);
        return std::nullopt;
    }

    template<auto R, typename V> static bool isValid(CSS::AnglePercentageRaw<R, V> raw, CSSPropertyParserOptions)
    {
        // Values other than 0 and +/-âˆž are not supported for <angle-percentage> numeric ranges currently.
        return isValidNonCanonicalizableDimensionValue(raw);
    }
};

template<auto R, typename V> struct ConsumerDefinition<CSS::AnglePercentage<R, V>> {
    using FunctionToken = FunctionConsumerForCalcValues<CSS::AnglePercentage<R, V>>;
    using DimensionToken = DimensionConsumer<CSS::AnglePercentage<R, V>, AnglePercentageValidator>;
    using PercentageToken = PercentageConsumer<CSS::AnglePercentage<R, V>, AnglePercentageValidator>;
    using NumberToken = NumberConsumerForUnitlessValues<CSS::AnglePercentage<R, V>, AnglePercentageValidator, CSS::AngleUnit::Deg>;
};

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
