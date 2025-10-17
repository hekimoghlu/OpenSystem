/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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

#include "CSSPrimitiveNumericTypes.h"

namespace WebCore {
namespace CSS {

// MARK: - Computed Style Dependencies

// What properties does this value rely on (eg, font-size for em units)?

// Most unit types have no dependencies.
template<UnitEnum Unit> struct ComputedStyleDependenciesCollector<Unit> {
    constexpr void operator()(ComputedStyleDependencies&, Unit)
    {
        // Nothing to do.
    }
};

// Let composite units dispatch to their component parts.
template<CompositeUnitEnum Unit> struct ComputedStyleDependenciesCollector<Unit> {
    constexpr void operator()(ComputedStyleDependencies& dependencies, Unit unit)
    {
        switchOnUnitType(unit, [&](auto unit) { collectComputedStyleDependencies(dependencies, unit); });
    }
};

// The one leaf unit type that does need to do work is `LengthUnit`.
template<> struct ComputedStyleDependenciesCollector<LengthUnit> {
    void operator()(ComputedStyleDependencies&, LengthUnit);
};

// Dependencies are based only on the unit; primitives to dispatch to the unit type analysis.
template<NumericRaw RawType> struct ComputedStyleDependenciesCollector<RawType> {
    constexpr void operator()(ComputedStyleDependencies& dependencies, const RawType& value)
    {
        collectComputedStyleDependencies(dependencies, value.unit);
    }
};

} // namespace CSS
} // namespace WebCore
