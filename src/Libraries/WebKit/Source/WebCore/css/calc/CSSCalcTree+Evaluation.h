/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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
#include "CSSToLengthConversionData.h"
#include <optional>

namespace WebCore {

namespace Calculation {
enum class Category : uint8_t;
}

class CSSCalcSymbolTable;

namespace CSSCalc {

struct Anchor;
struct Tree;

struct EvaluationOptions {
    // `category` represents the context in which the evaluation is taking place.
    Calculation::Category category;

    // `range` represents the allowed numeric range for the calculated result.
    CSS::Range range;

    // `conversionData` contains information needed to convert units into their canonical forms.
    std::optional<CSSToLengthConversionData> conversionData;

    // `symbolTable` contains information needed to convert unresolved symbols into numeric values.
    const CSSCalcSymbolTable& symbolTable;
};

std::optional<double> evaluateDouble(const Tree&, const EvaluationOptions&);
std::optional<double> evaluateWithoutFallback(const Anchor&, const EvaluationOptions&);

} // namespace CSSCalc
} // namespace WebCore
