/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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
#include <JavaScriptCore/JSONObject.h>

namespace WebCore {

template<> struct Converter<IDLJSON> : DefaultConverter<IDLJSON> {
    using Result = ConversionResult<IDLJSON>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value)
    {
        auto& vm = lexicalGlobalObject.vm();
        auto throwScope = DECLARE_THROW_SCOPE(vm);

        auto conversionResult = JSC::JSONStringify(&lexicalGlobalObject, value, 0);

        RETURN_IF_EXCEPTION(throwScope, Result::exception());

        return Result { WTFMove(conversionResult) };
    }
};

template<> struct JSConverter<IDLJSON> {
    static constexpr bool needsState = true;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(JSC::JSGlobalObject& lexicalGlobalObject, const String& value)
    {
        return JSC::JSONParse(&lexicalGlobalObject, value);
    }
};

} // namespace WebCore
