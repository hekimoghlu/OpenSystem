/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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

#include "CSSPrimitiveNumericRange.h"
#include "CSSValueConcepts.h"
#include <concepts>
#include <optional>
#include <wtf/Brigand.h>

namespace WebCore {

enum class CSSUnitType : uint8_t;

namespace Calculation {
enum class Category : uint8_t;
}

namespace CSS {

template<typename> struct UnitTraits;

// Concept covering all unit types.
template<typename T> concept UnitEnum = std::is_enum_v<T> && requires {
    requires std::integral<decltype(UnitTraits<T>::count)>;
    requires std::same_as<decltype(UnitTraits<T>::canonical), const T>;
    requires std::same_as<decltype(UnitTraits<T>::category), const Calculation::Category>;
    { UnitTraits<T>::validate(std::declval<CSSUnitType>()) } -> std::same_as<std::optional<T>>;
};

// Concept covering unit types where the enumeration contains only a single value.
//   e.g. IntegerUnit, NumberUnit, PercentageUnit, FlexUnit
template<typename T> concept SingleValueUnitEnum = UnitEnum<T> &&  requires {
    requires (UnitTraits<T>::count == 1);
};

// Concept covering unit types where the type is a composite of multiple other unit types.
//   e.g. AnglePercentageUnit, LengthPercentageUnit
template<typename T> concept CompositeUnitEnum = UnitEnum<T> && requires {
    typename UnitTraits<T>::Composite;
};

// Concept to check if a type, `T`, is one of the types `CompositeParent` composes over.
//   e.g. `NestedUnitEnumOf<LengthUnit, LengthPercentageUnit> == true`
template<typename T, typename CompositeParent> concept NestedUnitEnumOf = UnitEnum<T>
    && CompositeUnitEnum<CompositeParent>
    && brigand::contains<typename UnitTraits<CompositeParent>::Composite, T>::value;

// Forward declaration of PrimitiveNumericRaw to needed to create a hard constraint for the NumericRaw concept below.
template<Range, UnitEnum, typename> struct PrimitiveNumericRaw;

// Concept for use in generic contexts to filter on raw numeric CSS types.
template<typename T> concept NumericRaw = std::derived_from<T, PrimitiveNumericRaw<T::range, typename T::UnitType, typename T::ResolvedValueType>>;

// Forward declaration of PrimitiveNumeric to needed to create a hard constraint for the Numeric concept below.
template<NumericRaw> struct PrimitiveNumeric;

// Concept for use in generic contexts to filter on all numeric CSS types.
template<typename T> concept Numeric = VariantLike<T> && std::derived_from<T, PrimitiveNumeric<typename T::Raw>>;

// Concept for use in generic contexts to filter on non-composite numeric CSS types.
template<typename T> concept NonCompositeNumeric = Numeric<T> && (!CompositeUnitEnum<typename T::UnitType>);

// Concept for use in generic contexts to filter on dimension-percentage numeric CSS types.
template<typename T> concept DimensionPercentageNumeric = Numeric<T> && CompositeUnitEnum<typename T::UnitType>;

// Forward declaration of UnevaluatedCalc to needed to create a hard constraint for the Calc concept below.
template<NumericRaw> struct UnevaluatedCalc;

// Concept for use in generic contexts to filter on UnevaluatedCalc CSS types.
template<typename T> concept Calc = std::same_as<T, UnevaluatedCalc<typename T::Raw>>;

} // namespace CSS
} // namespace WebCore
