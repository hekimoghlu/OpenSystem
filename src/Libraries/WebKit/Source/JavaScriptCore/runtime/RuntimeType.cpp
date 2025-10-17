/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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
#include "RuntimeType.h"

#include "JSCJSValueInlines.h"

namespace JSC {

RuntimeType runtimeTypeForValue(JSValue value)
{
    if (UNLIKELY(!value))
        return TypeNothing;

    if (value.isUndefined())
        return TypeUndefined;
    if (value.isNull())
        return TypeNull;
    if (value.isAnyInt())
        return TypeAnyInt;
    if (value.isNumber())
        return TypeNumber;
    if (value.isString())
        return TypeString;
    if (value.isBoolean())
        return TypeBoolean;
    if (value.isObject())
        return TypeObject;
    if (value.isCallable())
        return TypeFunction;
    if (value.isSymbol())
        return TypeSymbol;
    if (value.isBigInt())
        return TypeBigInt;

    return TypeNothing;
}

String runtimeTypeAsString(RuntimeType type)
{
    if (type == TypeUndefined)
        return "Undefined"_s;
    if (type == TypeNull)
        return "Null"_s;
    if (type == TypeAnyInt)
        return "Integer"_s;
    if (type == TypeNumber)
        return "Number"_s;
    if (type == TypeString)
        return "String"_s;
    if (type == TypeObject)
        return "Object"_s;
    if (type == TypeBoolean)
        return "Boolean"_s;
    if (type == TypeFunction)
        return "Function"_s;
    if (type == TypeSymbol)
        return "Symbol"_s;
    if (type == TypeBigInt)
        return "BigInt"_s;
    if (type == TypeNothing)
        return "(Nothing)"_s;

    RELEASE_ASSERT_NOT_REACHED();
    return emptyString();
}

} // namespace JSC
