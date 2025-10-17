/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
#include "WebAssemblyStructConstructor.h"

#if ENABLE(WEBASSEMBLY)

#include "IteratorOperations.h"
#include "JSCInlines.h"
#include "JSWebAssemblyHelpers.h"
#include "JSWebAssemblyStruct.h"
#include "WebAssemblyStructPrototype.h"

namespace JSC {

const ClassInfo WebAssemblyStructConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyStructConstructor) };

static JSC_DECLARE_HOST_FUNCTION(constructJSWebAssemblyStruct);
static JSC_DECLARE_HOST_FUNCTION(callJSWebAssemblyStruct);

JSC_DEFINE_HOST_FUNCTION(constructJSWebAssemblyStruct, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    UNUSED_PARAM(callFrame);
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return throwVMTypeError(globalObject, scope, "WebAssembly.Struct is not accessible from JS"_s);
}

JSC_DEFINE_HOST_FUNCTION(callJSWebAssemblyStruct, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return throwVMTypeError(globalObject, scope, "WebAssembly.Struct is not accessible from JS"_s);
}

WebAssemblyStructConstructor* WebAssemblyStructConstructor::create(VM& vm, Structure* structure, WebAssemblyStructPrototype* thisPrototype)
{
    auto* constructor = new (NotNull, allocateCell<WebAssemblyStructConstructor>(vm)) WebAssemblyStructConstructor(vm, structure);
    constructor->finishCreation(vm, thisPrototype);
    return constructor;
}

Structure* WebAssemblyStructConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

void WebAssemblyStructConstructor::finishCreation(VM& vm, WebAssemblyStructPrototype* prototype)
{
    Base::finishCreation(vm, 1, "Struct"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum | PropertyAttribute::DontDelete);
}

WebAssemblyStructConstructor::WebAssemblyStructConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callJSWebAssemblyStruct, constructJSWebAssemblyStruct)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
