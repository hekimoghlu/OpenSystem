/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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
#include "CSSPrimitiveValue.h"
#include "CSSPrimitiveValueMappings.h"
#include "CSSValuePair.h"
#include "CSSValuePool.h"

namespace WebCore {
namespace CSS {

// MARK: - Conversion from strongly typed `CSS::` value types to `WebCore::CSSValue` types.

template<typename CSSType> struct CSSValueCreation;

template<typename CSSType> Ref<CSSValue> createCSSValue(const CSSType& value)
{
    return CSSValueCreation<CSSType>{}(value);
}

template<NumericRaw T> struct CSSValueCreation<T> {
    Ref<CSSValue> operator()(const T& raw)
    {
        return CSSPrimitiveValue::create(raw.value, toCSSUnitType(raw.unit));
    }
};

template<Calc T> struct CSSValueCreation<T> {
    Ref<CSSValue> operator()(const T& calc)
    {
        return CSSPrimitiveValue::create(calc.protectedCalc());
    }
};

template<Numeric T> struct CSSValueCreation<T> {
    Ref<CSSValue> operator()(const T& value)
    {
        return WTF::switchOn(value,
            [](const typename T::Raw& raw) {
                return CSSPrimitiveValue::create(raw.value, toCSSUnitType(raw.unit));
            },
            [](const typename T::Calc& calc) {
                return CSSPrimitiveValue::create(calc.protectedCalc());
            }
        );
    }
};

template<typename T> struct CSSValueCreation<SpaceSeparatedPoint<T>> {
    Ref<CSSValue> operator()(const SpaceSeparatedPoint<T>& value)
    {
        return CSSValuePair::create(
            WebCore::CSS::createCSSValue(value.x()),
            WebCore::CSS::createCSSValue(value.y())
        );
    }
};

template<typename T> struct CSSValueCreation<SpaceSeparatedSize<T>> {
    Ref<CSSValue> operator()(const SpaceSeparatedSize<T>& value)
    {
        return CSSValuePair::create(
            WebCore::CSS::createCSSValue(value.width()),
            WebCore::CSS::createCSSValue(value.height())
        );
    }
};

} // namespace CSS
} // namespace WebCore
