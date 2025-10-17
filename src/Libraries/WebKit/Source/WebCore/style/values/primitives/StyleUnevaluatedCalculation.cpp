/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#include "config.h"
#include "StyleUnevaluatedCalculation.h"

#include "CalculationValue.h"

namespace WebCore {
namespace Style {

UnevaluatedCalculationBase::UnevaluatedCalculationBase(Ref<CalculationValue>&& value)
    : calc { WTFMove(value) }
{
}

UnevaluatedCalculationBase::UnevaluatedCalculationBase(Calculation::Child&& root, Calculation::Category category, CSS::Range range)
    : calc {
        CalculationValue::create(
            category,
            Calculation::Range { range.min, range.max },
            Calculation::Tree { WTFMove(root) }
        )
    }
{
}

UnevaluatedCalculationBase::UnevaluatedCalculationBase(const UnevaluatedCalculationBase&) = default;
UnevaluatedCalculationBase::UnevaluatedCalculationBase(UnevaluatedCalculationBase&&) = default;
UnevaluatedCalculationBase& UnevaluatedCalculationBase::operator=(const UnevaluatedCalculationBase&) = default;
UnevaluatedCalculationBase& UnevaluatedCalculationBase::operator=(UnevaluatedCalculationBase&&) = default;

UnevaluatedCalculationBase::~UnevaluatedCalculationBase() = default;

Ref<CalculationValue> UnevaluatedCalculationBase::protectedCalculation() const
{
    return calc;
}

bool UnevaluatedCalculationBase::equal(const UnevaluatedCalculationBase& other) const
{
    return calc == other.calc;
}

} // namespace CSS
} // namespace WebCore
