/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
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
#include <JavaScriptCore/JSGlobalObject.h>
#include <wtf/WallTime.h>

namespace WebCore {

JSC::JSValue jsDate(JSC::JSGlobalObject&, WallTime value);
WallTime valueToDate(JSC::JSGlobalObject&, JSC::JSValue); // NaN if the value can't be converted to a date.

template<> struct Converter<IDLDate> : DefaultConverter<IDLDate> {
    using Result = ConversionResult<IDLDate>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        auto& vm = lexicalGlobalObject.vm();
        auto scope = DECLARE_THROW_SCOPE(vm);

        auto result = valueToDate(lexicalGlobalObject, value);
        RETURN_IF_EXCEPTION(scope, Result::exception());

        return Result { WTFMove(result) };
    }
};

template<> struct JSConverter<IDLDate> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, WallTime value)
    {
        return jsDate(lexicalGlobalObject, value);
    }
};

} // namespace WebCore
