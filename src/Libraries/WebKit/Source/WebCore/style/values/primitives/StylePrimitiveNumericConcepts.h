/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 16, 2025.
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

#include "CSSPrimitiveNumericConcepts.h"

namespace WebCore {
namespace Style {

// Forward declaration of PrimitiveNumeric to needed to create a hard constraint for the Numeric concept below.
template<CSS::Numeric> struct PrimitiveNumeric;

// Concept for use in generic contexts to filter on all numeric Style types.
template<typename T> concept Numeric = std::derived_from<T, PrimitiveNumeric<typename T::CSS>>;

// Concept for use in generic contexts to filter on non-composite numeric Style types.
template<typename T> concept NonCompositeNumeric = Numeric<T> && CSS::NonCompositeNumeric<typename T::CSS>;

// Concept for use in generic contexts to filter on dimension-percentage numeric Style types.
template<typename T> concept DimensionPercentageNumeric = Numeric<T> && VariantLike<T> && CSS::DimensionPercentageNumeric<typename T::CSS>;

// Forward declaration of UnevaluatedCalculation to needed to create a hard constraint for the Calc concept below.
template<CSS::Numeric> struct UnevaluatedCalculation;

// Concept for use in generic contexts to filter on UnevaluatedCalc Style types.
template<typename T> concept Calc = std::same_as<T, UnevaluatedCalculation<typename T::CSS>>;

} // namespace Style
} // namespace WebCore
