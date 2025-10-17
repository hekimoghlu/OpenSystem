/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 20, 2023.
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

#include "CSSCalcValue.h"
#include "CSSPrimitiveNumericTypes.h"
#include "CSSPrimitiveValue.h"
#include "CSSPropertyParserConsumer+MetaConsumer.h"
#include "CSSPropertyParserConsumer+MetaResolver.h"
#include "CSSPropertyParserOptions.h"
#include "CSSUnevaluatedCalc.h"
#include <wtf/RefPtr.h>

namespace WebCore {
namespace CSSPropertyParserHelpers {

// MARK: Resolver for users that want to get a CSSPrimitiveValue result.

/// Non-template base type for code sharing.
struct CSSPrimitiveValueResolverBase {
    static RefPtr<CSSPrimitiveValue> resolve(CSS::NumericRaw auto value, CSSPropertyParserOptions)
    {
        return CSSPrimitiveValue::create(value.value, CSS::toCSSUnitType(value.unit));
    }

    template<CSS::Range R, typename IntType> static RefPtr<CSSPrimitiveValue> resolve(CSS::IntegerRaw<R, IntType> value, CSSPropertyParserOptions)
    {
        return CSSPrimitiveValue::createInteger(value.value);
    }

    static RefPtr<CSSPrimitiveValue> resolve(CSS::Calc auto value, CSSPropertyParserOptions)
    {
        return CSSPrimitiveValue::create(value.protectedCalc());
    }

    static RefPtr<CSSPrimitiveValue> resolve(CSS::Numeric auto value, CSSPropertyParserOptions options)
    {
        return WTF::switchOn(WTFMove(value), [&](auto&& value) { return resolve(WTFMove(value), options); });
    }
};

template<typename... Ts>
struct CSSPrimitiveValueResolver : MetaResolver<RefPtr<CSSPrimitiveValue>, CSSPrimitiveValueResolverBase, Ts...> {
    using MetaResolver<RefPtr<CSSPrimitiveValue>, CSSPrimitiveValueResolverBase, Ts...>::resolve;
    using MetaResolver<RefPtr<CSSPrimitiveValue>, CSSPrimitiveValueResolverBase, Ts...>::consumeAndResolve;
};

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
