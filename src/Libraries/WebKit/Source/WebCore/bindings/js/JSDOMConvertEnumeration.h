/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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
#include <wtf/text/WTFString.h>

namespace WebCore {

// Specialized by generated code for IDL enumeration conversion.
template<typename T> std::optional<T> parseEnumerationFromString(const String&);
template<typename T> std::optional<T> parseEnumeration(JSC::JSGlobalObject&, JSC::JSValue);
template<typename T> ASCIILiteral expectedEnumerationValues();

// Specialized by generated code for IDL enumeration conversion.
template<typename T> JSC::JSString* convertEnumerationToJS(JSC::VM&, T);

template<typename T> struct Converter<IDLEnumeration<T>> : DefaultConverter<IDLEnumeration<T>> {
    using Result = ConversionResult<IDLEnumeration<T>>;

    template<typename ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, ExceptionThrower&& exceptionThrower = ExceptionThrower())
    {
        auto& vm = JSC::getVM(&lexicalGlobalObject);
        auto throwScope = DECLARE_THROW_SCOPE(vm);

        auto result = parseEnumeration<T>(lexicalGlobalObject, value);
        RETURN_IF_EXCEPTION(throwScope, Result::exception());

        if (UNLIKELY(!result)) {
            exceptionThrower(lexicalGlobalObject, throwScope);
            return Result::exception();
        }
        return Result { WTFMove(result.value()) };
    }
};

template<typename T> struct JSConverter<IDLEnumeration<T>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, T value)
    {
        return convertEnumerationToJS(lexicalGlobalObject.vm(), value);
    }
};

} // namespace WebCore
