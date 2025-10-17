/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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
#include "JSDOMConvertStrings.h"
#include "ScheduledAction.h"

namespace WebCore {

template<> struct Converter<IDLScheduledAction> : DefaultConverter<IDLScheduledAction> {
    using Result = ConversionResult<IDLScheduledAction>;

    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, JSDOMGlobalObject& globalObject, const String& sink)
    {
        JSC::VM& vm = JSC::getVM(&lexicalGlobalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);

        if (!value.isCallable()) {
            auto code = Converter<IDLStringContextTrustedScriptAdaptor<IDLDOMString>>::convert(lexicalGlobalObject, value, sink);
            RETURN_IF_EXCEPTION(scope, Result::exception());
            return ScheduledAction::create(globalObject.world(), code.releaseReturnValue());
        }

        // The value must be an object at this point because no non-object values are callable.
        ASSERT(value.isObject());
        return ScheduledAction::create(globalObject.world(), JSC::Strong<JSC::JSObject> { vm, JSC::asObject(value) });
    }
};

}

