/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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

#include "JSCJSValue.h"
#include "TypedArrayAdaptors.h"

namespace JSC {

template<typename Adaptor>
typename Adaptor::Type toNativeFromValue(JSValue value)
{
    ASSERT(!value.isBigInt());
    if (value.isInt32())
        return Adaptor::toNativeFromInt32(value.asInt32());
    return Adaptor::toNativeFromDouble(value.asDouble());
}

template<typename Adaptor>
typename Adaptor::Type toNativeFromValue(JSGlobalObject* globalObject, JSValue value)
{
    if constexpr (std::is_same_v<Adaptor, BigInt64Adaptor> || std::is_same_v<Adaptor, BigUint64Adaptor>) {
        if constexpr (std::is_same_v<Adaptor, BigInt64Adaptor>)
            return value.toBigInt64(globalObject);
        else
            return value.toBigUInt64(globalObject);
    } else {
        if (value.isInt32())
            return Adaptor::toNativeFromInt32(value.asInt32());
        if (value.isNumber())
            return Adaptor::toNativeFromDouble(value.asDouble());
        return Adaptor::toNativeFromDouble(value.toNumber(globalObject));
    }
}

template<typename Adaptor>
std::optional<typename Adaptor::Type> toNativeFromValueWithoutCoercion(JSValue value)
{
    if constexpr (std::is_same_v<Adaptor, BigInt64Adaptor> || std::is_same_v<Adaptor, BigUint64Adaptor>) {
        if (!value.isBigInt())
            return std::nullopt;
        if constexpr (std::is_same_v<Adaptor, BigInt64Adaptor>)
            return JSBigInt::toBigInt64(value);
        else
            return JSBigInt::toBigUInt64(value);
    } else {
        if (!value.isNumber())
            return std::nullopt;
        if (value.isInt32())
            return Adaptor::toNativeFromInt32WithoutCoercion(value.asInt32());
        return Adaptor::toNativeFromDoubleWithoutCoercion(value.asDouble());
    }
}

} // namespace JSC
