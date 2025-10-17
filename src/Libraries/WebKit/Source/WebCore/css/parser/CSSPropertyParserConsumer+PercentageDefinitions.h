/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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

struct PercentageValidator {
    static constexpr std::optional<CSS::PercentageUnit> validate(CSSUnitType unitType, CSSPropertyParserOptions)
    {
        return CSS::UnitTraits<CSS::PercentageUnit>::validate(unitType);
    }

    template<auto R, typename V> static bool isValid(CSS::PercentageRaw<R, V> raw, CSSPropertyParserOptions)
    {
        return isValidCanonicalValue(raw);
    }
};

template<auto R, typename V> struct ConsumerDefinition<CSS::Percentage<R, V>> {
    using FunctionToken = FunctionConsumerForCalcValues<CSS::Percentage<R, V>>;
    using PercentageToken = PercentageConsumer<CSS::Percentage<R, V>, PercentageValidator>;
};

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
