/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
#include "JSValueInWrappedObject.h"

namespace WebCore {

template<> struct Converter<IDLAny> : DefaultConverter<IDLAny> {
    static constexpr bool conversionHasSideEffects = false;

    static ConversionResult<IDLAny> convert(JSC::JSGlobalObject&, JSC::JSValue value)
    {
        return value;
    }

    static ConversionResult<IDLAny> convert(const JSC::Strong<JSC::Unknown>& value)
    {
        return value.get();
    }
};

template<> struct JSConverter<IDLAny> {
    static constexpr bool needsState = false;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(const JSC::JSValue& value)
    {
        return value;
    }

    static JSC::JSValue convert(const JSC::Strong<JSC::Unknown>& value)
    {
        return value.get();
    }

    static JSC::JSValue convert(const JSValueInWrappedObject& value)
    {
        return value.getValue();
    }
};

template<> struct VariadicConverter<IDLAny> {
    using Item = JSC::Strong<JSC::Unknown>;

    static std::optional<Item> convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        return Item { JSC::getVM(&lexicalGlobalObject), value };
    }
};

} // namespace WebCore
