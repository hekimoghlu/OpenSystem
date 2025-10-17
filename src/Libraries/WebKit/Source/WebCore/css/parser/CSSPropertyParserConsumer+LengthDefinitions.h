/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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

#include "CSSPropertyParserConsumer+MetaConsumerDefinitions.h"

namespace WebCore {
namespace CSSPropertyParserHelpers {

struct LengthValidator {
    static constexpr std::optional<CSS::LengthUnit> validate(CSSUnitType unitType, CSSPropertyParserOptions options)
    {
        if (unitType == CSSUnitType::CSS_QUIRKY_EM && !isUASheetBehavior(options.parserMode))
            return std::nullopt;
        return CSS::UnitTraits<CSS::LengthUnit>::validate(unitType);
    }

    template<auto R, typename V> static bool isValid(CSS::LengthRaw<R, V> raw, CSSPropertyParserOptions)
    {
        // Values other than 0 and +/-âˆž are not supported for <length> numeric ranges currently.
        return isValidNonCanonicalizableDimensionValue(raw);
    }
};

template<auto R, typename V> struct ConsumerDefinition<CSS::Length<R, V>> {
    using FunctionToken = FunctionConsumerForCalcValues<CSS::Length<R, V>>;
    using DimensionToken = DimensionConsumer<CSS::Length<R, V>, LengthValidator>;
    using NumberToken = NumberConsumerForUnitlessValues<CSS::Length<R, V>, LengthValidator, CSS::LengthUnit::Px>;
};

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
