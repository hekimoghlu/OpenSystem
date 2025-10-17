/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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

#include "JSDOMConvertDictionary.h"
#include "JSDOMConvertNullable.h"

namespace WebCore {

namespace Detail {

template<typename IDL>
struct OptionalConversionType;

template<typename IDL>
struct OptionalConversionType {
    using Type = typename IDLOptional<IDL>::ConversionResultType;
};

template<>
struct OptionalConversionType<IDLObject> {
    using Type = std::optional<JSC::Strong<JSC::JSObject>>;
};

template<typename T>
struct OptionalConversionType<IDLDictionary<T>> {
    using Type = std::conditional_t<std::is_default_constructible_v<T>, T, std::optional<T>>;
};

}

// `IDLOptional` is just like `IDLNullable`, but used in places that where the type is implicitly optional,
// like optional arguments to functions without default values, or non-required members of dictionaries
// without default values.
//
// As such, rather than checking `isUndefinedOrNull()`, IDLOptional uses `isUndefined()` matching what
// is needed in those cases.

template<typename IDL> struct Converter<IDLOptional<IDL>> : DefaultConverter<IDLOptional<IDL>> {
    using ReturnType = typename Detail::OptionalConversionType<IDL>::Type;
    using Result = ConversionResult<IDLOptional<IDL>>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        if (value.isUndefined())
            return ReturnType { };
        return WebCore::convert<IDL>(lexicalGlobalObject, value);
    }
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSC::JSObject& thisObject)
    {
        if (value.isUndefined())
            return ReturnType { };
        return WebCore::convert<IDL>(lexicalGlobalObject, value, thisObject);
    }
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSDOMGlobalObject& globalObject)
    {
        if (value.isUndefined())
            return ReturnType { };
        return WebCore::convert<IDL>(lexicalGlobalObject, value, globalObject);
    }
    template<ExceptionThrowerFunctor ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, ExceptionThrower&& exceptionThrower)
    {
        if (value.isUndefined())
            return ReturnType { };
        return WebCore::convert<IDL>(lexicalGlobalObject, value, std::forward<ExceptionThrower>(exceptionThrower));
    }
    template<ExceptionThrowerFunctor ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSC::JSObject& thisObject, ExceptionThrower&& exceptionThrower)
    {
        if (value.isUndefined())
            return ReturnType { };
        return WebCore::convert<IDL>(lexicalGlobalObject, value, thisObject, std::forward<ExceptionThrower>(exceptionThrower));
    }
    template<ExceptionThrowerFunctor ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSDOMGlobalObject& globalObject, ExceptionThrower&& exceptionThrower)
    {
        if (value.isUndefined())
            return ReturnType { };
        return WebCore::convert<IDL>(lexicalGlobalObject, value, globalObject, std::forward<ExceptionThrower>(exceptionThrower));
    }
};

// MARK: Helper functions for invoking an optional conversion.

template<typename IDL, DefaultValueFunctor<IDL> DefaultValueFunctor>
ConversionResult<IDL> convertOptionalWithDefault(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, DefaultValueFunctor&& defaultValue)
{
    if (value.isUndefined())
        return defaultValue();
    return convert<IDL>(lexicalGlobalObject, value);
}

template<typename IDL, DefaultValueFunctor<IDL> DefaultValueFunctor>
ConversionResult<IDL> convertOptionalWithDefault(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSC::JSObject& thisObject, DefaultValueFunctor&& defaultValue)
{
    if (value.isUndefined())
        return defaultValue();
    return convert<IDL>(lexicalGlobalObject, value, thisObject);
}

template<typename IDL, DefaultValueFunctor<IDL> DefaultValueFunctor>
ConversionResult<IDL> convertOptionalWithDefault(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSDOMGlobalObject& globalObject, DefaultValueFunctor&& defaultValue)
{
    if (value.isUndefined())
        return defaultValue();
    return convert<IDL>(lexicalGlobalObject, value, globalObject);
}

template<typename IDL, DefaultValueFunctor<IDL> DefaultValueFunctor, ExceptionThrowerFunctor ExceptionThrower>
ConversionResult<IDL> convertOptionalWithDefault(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, DefaultValueFunctor&& defaultValue, ExceptionThrower&& exceptionThrower)
{
    if (value.isUndefined())
        return defaultValue();
    return convert<IDL>(lexicalGlobalObject, value, std::forward<ExceptionThrower>(exceptionThrower));
}

template<typename IDL, DefaultValueFunctor<IDL> DefaultValueFunctor, ExceptionThrowerFunctor ExceptionThrower>
ConversionResult<IDL> convertOptionalWithDefault(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSC::JSObject& thisObject, DefaultValueFunctor&& defaultValue, ExceptionThrower&& exceptionThrower)
{
    if (value.isUndefined())
        return defaultValue();
    return convert<IDL>(lexicalGlobalObject, value, thisObject, std::forward<ExceptionThrower>(exceptionThrower));
}

template<typename IDL, DefaultValueFunctor<IDL> DefaultValueFunctor, ExceptionThrowerFunctor ExceptionThrower>
ConversionResult<IDL> convertOptionalWithDefault(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSDOMGlobalObject& globalObject, DefaultValueFunctor&& defaultValue, ExceptionThrower&& exceptionThrower)
{
    if (value.isUndefined())
        return defaultValue();
    return convert<IDL>(lexicalGlobalObject, value, globalObject, std::forward<ExceptionThrower>(exceptionThrower));
}

} // namespace WebCore
