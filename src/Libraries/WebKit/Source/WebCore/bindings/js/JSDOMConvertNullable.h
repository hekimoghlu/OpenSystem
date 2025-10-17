/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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

#include "IDLTypes.h"
#include "JSDOMConvertAny.h"
#include "JSDOMConvertInterface.h"
#include "JSDOMConvertNumbers.h"
#include "JSDOMConvertStrings.h"

namespace WebCore {

template<typename IDL> struct Converter<IDLNullable<IDL>> : DefaultConverter<IDLNullable<IDL>> {
    using Result = ConversionResult<IDLNullable<IDL>>;

    // 1. If Type(V) is not Object, and the conversion to an IDL value is being performed
    // due to V being assigned to an attribute whose type is a nullable callback function
    // that is annotated with [LegacyTreatNonObjectAsNull], then return the IDL nullable
    // type T? value null.
    //
    // NOTE: Handled elsewhere.
    //
    // 2. Otherwise, if V is null or undefined, then return the IDL nullable type T? value null.
    // 3. Otherwise, return the result of converting V using the rules for the inner IDL type T.

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        if (value.isUndefinedOrNull())
            return { IDL::nullValue() };
        return WebCore::convert<IDL>(lexicalGlobalObject, value);
    }
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSC::JSObject& thisObject)
    {
        if (value.isUndefinedOrNull())
            return { IDL::nullValue() };
        return WebCore::convert<IDL>(lexicalGlobalObject, value, thisObject);
    }
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSDOMGlobalObject& globalObject)
    {
        if (value.isUndefinedOrNull())
            return { IDL::nullValue() };
        return WebCore::convert<IDL>(lexicalGlobalObject, value, globalObject);
    }
    template<typename ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, ExceptionThrower&& exceptionThrower)
    {
        if (value.isUndefinedOrNull())
            return { IDL::nullValue() };
        return WebCore::convert<IDL>(lexicalGlobalObject, value, std::forward<ExceptionThrower>(exceptionThrower));
    }
    template<typename ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSC::JSObject& thisObject, ExceptionThrower&& exceptionThrower)
    {
        if (value.isUndefinedOrNull())
            return { IDL::nullValue() };
        return WebCore::convert<IDL>(lexicalGlobalObject, value, thisObject, std::forward<ExceptionThrower>(exceptionThrower));
    }
    template<typename ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSDOMGlobalObject& globalObject, ExceptionThrower&& exceptionThrower)
    {
        if (value.isUndefinedOrNull())
            return { IDL::nullValue() };
        return WebCore::convert<IDL>(lexicalGlobalObject, value, globalObject, std::forward<ExceptionThrower>(exceptionThrower));
    }
};

template<typename IDL> struct JSConverter<IDLNullable<IDL>> {
    using ImplementationType = typename IDLNullable<IDL>::ImplementationType;

    static constexpr bool needsState = JSConverter<IDL>::needsState;
    static constexpr bool needsGlobalObject = JSConverter<IDL>::needsGlobalObject;

    template<std::convertible_to<ImplementationType> U>
    static JSC::JSValue convert(U&& value)
    {
        if (IDL::isNullValue(value))
            return JSC::jsNull();
        return toJS<IDL>(IDL::extractValueFromNullable(std::forward<U>(value)));
    }

    template<std::convertible_to<ImplementationType> U>
    static JSC::JSValue convert(const U& value)
    {
        if (IDL::isNullValue(value))
            return JSC::jsNull();
        return toJS<IDL>(IDL::extractValueFromNullable(value));
    }

    template<std::convertible_to<ImplementationType> U>
    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, U&& value)
    {
        if (IDL::isNullValue(value))
            return JSC::jsNull();
        return toJS<IDL>(lexicalGlobalObject, IDL::extractValueFromNullable(std::forward<U>(value)));
    }

    template<std::convertible_to<ImplementationType> U>
    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const U& value)
    {
        if (IDL::isNullValue(value))
            return JSC::jsNull();
        return toJS<IDL>(lexicalGlobalObject, IDL::extractValueFromNullable(value));
    }

    template<std::convertible_to<ImplementationType> U>
    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, U&& value)
    {
        if (IDL::isNullValue(value))
            return JSC::jsNull();
        return toJS<IDL>(lexicalGlobalObject, globalObject, IDL::extractValueFromNullable(std::forward<U>(value)));
    }

    template<std::convertible_to<ImplementationType> U>
    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, const U& value)
    {
        if (IDL::isNullValue(value))
            return JSC::jsNull();
        return toJS<IDL>(lexicalGlobalObject, globalObject, IDL::extractValueFromNullable(value));
    }

    template<std::convertible_to<ImplementationType> U>
    static JSC::JSValue convertNewlyCreated(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, U&& value)
    {
        if (IDL::isNullValue(value))
            return JSC::jsNull();
        return toJSNewlyCreated<IDL>(lexicalGlobalObject, globalObject, IDL::extractValueFromNullable(std::forward<U>(value)));
    }
};

} // namespace WebCore
