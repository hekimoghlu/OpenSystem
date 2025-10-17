/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#include "JSDOMConvertBase.h"
#include "JSDOMGlobalObject.h"

namespace WebCore {

template<typename ImplementationClass> struct JSDOMCallbackConverterTraits;

// An example of an implementation of the traits.
//
// template<> struct JSDOMCallbackConverterTraits<JSNodeFilter> {
//     using Base = NodeFilter;
// };
//
// These will be produced by the code generator.

template<typename T> struct Converter<IDLCallbackFunction<T>> : DefaultConverter<IDLCallbackFunction<T>> {
    static constexpr bool conversionHasSideEffects = false;

    using Result = ConversionResult<IDLCallbackFunction<T>>;

    template<typename ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSDOMGlobalObject& globalObject, ExceptionThrower&& exceptionThrower = ExceptionThrower())
    {
        JSC::VM& vm = JSC::getVM(&lexicalGlobalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);

        if (!value.isCallable()) {
            exceptionThrower(lexicalGlobalObject, scope);
            return Result::exception();
        }

        return Result { T::create(JSC::asObject(value), &globalObject) };
    }
};

template<typename T> struct JSConverter<IDLCallbackFunction<T>> {
    static constexpr bool needsState = false;
    static constexpr bool needsGlobalObject = false;

    using Base = typename JSDOMCallbackConverterTraits<T>::Base;

    static JSC::JSValue convert(const Base& value)
    {
        return toJS(Detail::getPtrOrRef(value));
    }

    static JSC::JSValue convert(const Ref<Base>& value)
    {
        return toJS(Detail::getPtrOrRef(value));
    }

    static JSC::JSValue convertNewlyCreated(Ref<Base>&& value)
    {
        return toJSNewlyCreated(std::forward<Base>(value));
    }
};

// Specialization of nullable callback function to account for unconventional base type input.
template<typename T> struct JSConverter<IDLNullable<IDLCallbackFunction<T>>> {
    static constexpr bool needsState = false;
    static constexpr bool needsGlobalObject = false;

    using Base = typename JSDOMCallbackConverterTraits<T>::Base;

    static JSC::JSValue convert(const Base* value)
    {
        if (!value)
            return JSC::jsNull();
        return toJS<IDLCallbackFunction<T>>(*value);
    }

    static JSC::JSValue convert(const RefPtr<Base>& value)
    {
        if (!value)
            return JSC::jsNull();
        return toJS<IDLCallbackFunction<T>>(*value);
    }

    static JSC::JSValue convertNewlyCreated(RefPtr<Base>&& value)
    {
        if (!value)
            return JSC::jsNull();
        return toJSNewlyCreated<IDLCallbackFunction<T>>(value.releaseNonNull());
    }
};

template<typename T> struct Converter<IDLCallbackInterface<T>> : DefaultConverter<IDLCallbackInterface<T>> {
    using Result = ConversionResult<IDLCallbackInterface<T>>;

    template<typename ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSDOMGlobalObject& globalObject, ExceptionThrower&& exceptionThrower = ExceptionThrower())
    {
        JSC::VM& vm = JSC::getVM(&lexicalGlobalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);

        if (!value.isObject()) {
            exceptionThrower(lexicalGlobalObject, scope);
            return Result::exception();
        }

        return Result { T::create(JSC::asObject(value), &globalObject) };
    }
};

template<typename T> struct JSConverter<IDLCallbackInterface<T>> {
    static constexpr bool needsState = false;
    static constexpr bool needsGlobalObject = false;

    using Base = typename JSDOMCallbackConverterTraits<T>::Base;

    static JSC::JSValue convert(const Base& value)
    {
        return toJS(Detail::getPtrOrRef(value));
    }

    static JSC::JSValue convert(const Ref<Base>& value)
    {
        return toJS(Detail::getPtrOrRef(value));
    }

    static JSC::JSValue convertNewlyCreated(Ref<Base>&& value)
    {
        return toJSNewlyCreated(std::forward<Base>(value));
    }
};

// Specialization of nullable callback interface to account for unconventional base type input.
template<typename T> struct JSConverter<IDLNullable<IDLCallbackInterface<T>>> {
    static constexpr bool needsState = false;
    static constexpr bool needsGlobalObject = false;

    using Base = typename JSDOMCallbackConverterTraits<T>::Base;

    static JSC::JSValue convert(const Base* value)
    {
        if (!value)
            return JSC::jsNull();
        return toJS<IDLCallbackInterface<T>>(*value);
    }

    static JSC::JSValue convert(const RefPtr<Base>& value)
    {
        if (!value)
            return JSC::jsNull();
        return toJS<IDLCallbackInterface<T>>(*value);
    }

    static JSC::JSValue convertNewlyCreated(RefPtr<Base>&& value)
    {
        if (!value)
            return JSC::jsNull();
        return toJSNewlyCreated<IDLCallbackInterface<T>>(value.releaseNonNull());
    }
};

} // namespace WebCore
