/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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

#include "JSCustomXPathNSResolver.h"
#include "JSXPathNSResolver.h"

namespace WebCore {

template<> struct IDLInterface<XPathNSResolver> : IDLWrapper<XPathNSResolver> {
    using ConversionResultType = Ref<XPathNSResolver>;
    using NullableConversionResultType = RefPtr<XPathNSResolver>;
};

template<> struct Converter<IDLInterface<XPathNSResolver>> : DefaultConverter<IDLInterface<XPathNSResolver>> {
    using Result = ConversionResult<IDLInterface<XPathNSResolver>>;

    template<typename ExceptionThrower = DefaultExceptionThrower>
    static Result convert(JSC::JSGlobalObject& lexicalGlobalObject, JSC::JSValue value, ExceptionThrower&& exceptionThrower = ExceptionThrower())
    {
        JSC::VM& vm = JSC::getVM(&lexicalGlobalObject);
        auto scope = DECLARE_THROW_SCOPE(vm);
        if (!value.isObject()) {
            exceptionThrower(lexicalGlobalObject, scope);
            return Result::exception();
        }

        auto object = asObject(value);
        if (object->inherits<JSXPathNSResolver>())
            return { JSC::jsCast<JSXPathNSResolver*>(object)->wrapped() };

        return { JSCustomXPathNSResolver::create(object, JSC::jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject)) };
    }
};

} // namespace WebCore
