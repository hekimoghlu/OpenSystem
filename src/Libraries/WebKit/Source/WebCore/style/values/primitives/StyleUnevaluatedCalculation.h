/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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

#include "StylePrimitiveNumericConcepts.h"
#include <wtf/Forward.h>

namespace WebCore {

namespace Calculation {
struct Child;
}

class CalculationValue;

namespace Style {

// Non-generic base type to allow code sharing and out-of-line definitions.
struct UnevaluatedCalculationBase {
    explicit UnevaluatedCalculationBase(Ref<CalculationValue>&&);
    explicit UnevaluatedCalculationBase(Calculation::Child&&, Calculation::Category, CSS::Range);

    WEBCORE_EXPORT UnevaluatedCalculationBase(const UnevaluatedCalculationBase&);
    WEBCORE_EXPORT UnevaluatedCalculationBase(UnevaluatedCalculationBase&&);
    UnevaluatedCalculationBase& operator=(const UnevaluatedCalculationBase&);
    UnevaluatedCalculationBase& operator=(UnevaluatedCalculationBase&&);

    WEBCORE_EXPORT ~UnevaluatedCalculationBase();

    Ref<CalculationValue> protectedCalculation() const;

    bool equal(const UnevaluatedCalculationBase&) const;

private:
    Ref<CalculationValue> calc;
};

// Wrapper for `Ref<CalculationValue>` that includes range and category as part of the type.
template<CSS::Numeric CSSType> struct UnevaluatedCalculation : UnevaluatedCalculationBase {
    using UnevaluatedCalculationBase::UnevaluatedCalculationBase;
    using UnevaluatedCalculationBase::operator=;

    using CSS = CSSType;
    static constexpr auto range = CSS::range;
    static constexpr auto category = CSS::category;

    explicit UnevaluatedCalculation(Calculation::Child&& child)
        : UnevaluatedCalculationBase(WTFMove(child), category, range)
    {
    }

    bool operator==(const UnevaluatedCalculation& other) const
    {
        return UnevaluatedCalculationBase::equal(static_cast<const UnevaluatedCalculationBase&>(other));
    }
};

} // namespace Style
} // namespace WebCore

namespace WTF {
template<WebCore::Style::Calc T> struct IsSmartPtr<T> {
    static constexpr bool value = true;
};

} // namespace WTF
