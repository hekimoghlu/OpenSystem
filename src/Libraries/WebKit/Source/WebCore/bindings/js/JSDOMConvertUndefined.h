/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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

template<> struct Converter<IDLUndefined> : DefaultConverter<IDLUndefined> {
    static constexpr bool conversionHasSideEffects = false;
    using Result = ConversionResult<IDLUndefined>;

    static Result convert(JSC::JSGlobalObject&, JSC::JSValue)
    {
        return Result { std::monostate { } };
    }
};

template<> struct JSConverter<IDLUndefined> {
    static constexpr bool needsState = false;
    static constexpr bool needsGlobalObject = false;

    static JSC::JSValue convert(std::monostate)
    {
        return JSC::jsUndefined();
    }
};

} // namespace WebCore
