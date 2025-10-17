/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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
#include "config.h"
#include "WebAssemblyArrayConstructor.h"

#if ENABLE(WEBASSEMBLY)

#include "JSCInlines.h"
#include "JSWebAssemblyArray.h"
#include "WebAssemblyArrayPrototype.h"

namespace JSC {

const ClassInfo WebAssemblyArrayConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyArrayConstructor) };

static JSC_DECLARE_HOST_FUNCTION(constructJSWebAssemblyArray);
static JSC_DECLARE_HOST_FUNCTION(callJSWebAssemblyArray);

JSC_DEFINE_HOST_FUNCTION(constructJSWebAssemblyArray, (JSGlobalObject* globalObject, CallFrame*))
{
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    return throwVMTypeError(globalObject, scope, "WebAssembly.Array constructor should not be exposed currently"_s);
}

JSC_DEFINE_HOST_FUNCTION(callJSWebAssemblyArray, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "WebAssembly.Array"_s));
}

WebAssemblyArrayConstructor* WebAssemblyArrayConstructor::create(VM& vm, Structure* structure, WebAssemblyArrayPrototype* thisPrototype)
{
    auto* constructor = new (NotNull, allocateCell<WebAssemblyArrayConstructor>(vm)) WebAssemblyArrayConstructor(vm, structure);
    constructor->finishCreation(vm, thisPrototype);
    return constructor;
}

Structure* WebAssemblyArrayConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

void WebAssemblyArrayConstructor::finishCreation(VM& vm, WebAssemblyArrayPrototype* prototype)
{
    Base::finishCreation(vm, 1, "Array"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum | PropertyAttribute::DontDelete);
}

WebAssemblyArrayConstructor::WebAssemblyArrayConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callJSWebAssemblyArray, constructJSWebAssemblyArray)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
