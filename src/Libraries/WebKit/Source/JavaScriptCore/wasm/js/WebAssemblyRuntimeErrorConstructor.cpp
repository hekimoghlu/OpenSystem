/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
#include "WebAssemblyRuntimeErrorConstructor.h"

#if ENABLE(WEBASSEMBLY)

#include "JSCInlines.h"
#include "JSWebAssemblyRuntimeError.h"
#include "WebAssemblyRuntimeErrorPrototype.h"

namespace JSC {

const ClassInfo WebAssemblyRuntimeErrorConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyRuntimeErrorConstructor) };

static JSC_DECLARE_HOST_FUNCTION(constructJSWebAssemblyRuntimeError);
static JSC_DECLARE_HOST_FUNCTION(callJSWebAssemblyRuntimeError);

JSC_DEFINE_HOST_FUNCTION(constructJSWebAssemblyRuntimeError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue message = callFrame->argument(0);
    JSValue options = callFrame->argument(1);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, webAssemblyRuntimeErrorStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(ErrorInstance::create(globalObject, structure, message, options, nullptr, TypeNothing, ErrorType::Error, false)));
}

JSC_DEFINE_HOST_FUNCTION(callJSWebAssemblyRuntimeError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    JSValue message = callFrame->argument(0);
    JSValue options = callFrame->argument(1);
    Structure* errorStructure = globalObject->webAssemblyRuntimeErrorStructure();
    return JSValue::encode(ErrorInstance::create(globalObject, errorStructure, message, options, nullptr, TypeNothing, ErrorType::Error, false));
}

WebAssemblyRuntimeErrorConstructor* WebAssemblyRuntimeErrorConstructor::create(VM& vm, Structure* structure, WebAssemblyRuntimeErrorPrototype* thisPrototype)
{
    auto* constructor = new (NotNull, allocateCell<WebAssemblyRuntimeErrorConstructor>(vm)) WebAssemblyRuntimeErrorConstructor(vm, structure);
    constructor->finishCreation(vm, thisPrototype);
    return constructor;
}

Structure* WebAssemblyRuntimeErrorConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

void WebAssemblyRuntimeErrorConstructor::finishCreation(VM& vm, WebAssemblyRuntimeErrorPrototype* prototype)
{
    Base::finishCreation(vm, 1, "RuntimeError"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum | PropertyAttribute::DontDelete);
}

WebAssemblyRuntimeErrorConstructor::WebAssemblyRuntimeErrorConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callJSWebAssemblyRuntimeError, constructJSWebAssemblyRuntimeError)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
