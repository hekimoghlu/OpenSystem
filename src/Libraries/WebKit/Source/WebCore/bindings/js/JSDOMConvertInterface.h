/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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
#include <JavaScriptCore/Error.h>

namespace WebCore {

template<typename ImplementationClass> struct JSDOMWrapperConverterTraits;

template<typename T, typename Enable = void>
struct JSToWrappedOverloader {
    using ReturnType = typename JSDOMWrapperConverterTraits<T>::ToWrappedReturnType;
    using WrapperType = typename JSDOMWrapperConverterTraits<T>::WrapperClass;

    static ReturnType toWrapped(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        return WrapperType::toWrapped(JSC::getVM(&lexicalGlobalObject), value);
    }
};

template<typename T>
struct JSToWrappedOverloader<T, typename std::enable_if<JSDOMWrapperConverterTraits<T>::needsState>::type> {
    using ReturnType = typename JSDOMWrapperConverterTraits<T>::ToWrappedReturnType;
    using WrapperType = typename JSDOMWrapperConverterTraits<T>::WrapperClass;

    static ReturnType toWrapped(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        return WrapperType::toWrapped(lexicalGlobalObject, value);
    }
};

template<typename T> struct Converter<IDLInterface<T>> : DefaultConverter<IDLInterface<T>> {
    using Result = ConversionResult<IDLInterface<T>>;

    template<typename ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, ExceptionThrower&& exceptionThrower = ExceptionThrower())
    {
        auto& vm = JSC::getVM(&lexicalGlobalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);

        auto object = JSToWrappedOverloader<T>::toWrapped(lexicalGlobalObject, value);
        if (UNLIKELY(!object)) {
            exceptionThrower(lexicalGlobalObject, scope);
            return Result::exception();
        }

        return Result { object };
    }
};

template<typename T> struct JSConverter<IDLInterface<T>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = true;

    template <typename U>
    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, const U& value)
    {
        return toJS(&lexicalGlobalObject, &globalObject, Detail::getPtrOrRef(value));
    }

    template<typename U>
    static JSC::JSValue convertNewlyCreated(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, U&& value)
    {
        return toJSNewlyCreated(&lexicalGlobalObject, &globalObject, std::forward<U>(value));
    }
};

template<typename T> struct VariadicConverter<IDLInterface<T>> {
    using Item = std::reference_wrapper<T>;

    static std::optional<Item> convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        auto& vm = lexicalGlobalObject.vm();
        auto scope = DECLARE_THROW_SCOPE(vm);

        auto result = WebCore::convert<IDLInterface<T>>(lexicalGlobalObject, value);
        if (UNLIKELY(result.hasException(scope)))
            return std::nullopt;

        return Item { *result.releaseReturnValue() };
    }
};

} // namespace WebCore
