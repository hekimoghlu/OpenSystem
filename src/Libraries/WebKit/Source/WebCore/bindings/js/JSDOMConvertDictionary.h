/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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

// Specialized by generated code for IDL dictionary conversion.
template<typename T> ConversionResult<IDLDictionary<T>> convertDictionary(JSC::JSGlobalObject&, JSC::JSValue);


template<typename T> struct Converter<IDLDictionary<T>> : DefaultConverter<IDLDictionary<T>> {
    using Result = ConversionResult<IDLDictionary<T>>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        return convertDictionary<T>(lexicalGlobalObject, value);
    }
};

template<typename T> struct JSConverter<IDLDictionary<T>> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = true;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, const T& dictionary)
    {
        return convertDictionaryToJS(lexicalGlobalObject, globalObject, dictionary);
    }
};

} // namespace WebCore
