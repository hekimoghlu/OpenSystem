/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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

namespace WebCore {

template<typename T> struct Converter<IDLEventListener<T>> : DefaultConverter<IDLEventListener<T>> {
    using Result = ConversionResult<IDLEventListener<T>>;

    template<typename ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSC::JSObject& thisObject, ExceptionThrower&& exceptionThrower = ExceptionThrower())
    {
        auto scope = DECLARE_THROW_SCOPE(JSC::getVM(&lexicalGlobalObject));

        if (UNLIKELY(!value.isObject())) {
            exceptionThrower(lexicalGlobalObject, scope);
            return Result::exception();
        }

        constexpr bool isAttribute = false;
        return Result { T::create(*asObject(value), thisObject, isAttribute, currentWorld(lexicalGlobalObject)) };
    }
};

} // namespace WebCore
