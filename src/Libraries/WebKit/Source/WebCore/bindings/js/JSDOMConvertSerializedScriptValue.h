/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 14, 2023.
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
#include "JSDOMConvertStrings.h"

namespace WebCore {

template<typename T> struct Converter<IDLSerializedScriptValue<T>> : DefaultConverter<IDLSerializedScriptValue<T>> {
    using Result = ConversionResult<IDLSerializedScriptValue<T>>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        auto result = T::convert(lexicalGlobalObject, value);
        if (!result)
            return Result::exception();

        return Result { result.releaseNonNull() };
    }
};

template<typename T> struct JSConverter<IDLSerializedScriptValue<T>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = true;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, RefPtr<T> value)
    {
        return value ? value->deserialize(lexicalGlobalObject, &globalObject) : JSC::jsNull();
    }
};

} // namespace WebCore
