/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#include "WebAssemblyCompileErrorConstructor.h"

#if ENABLE(WEBASSEMBLY)

#include "JSCInlines.h"
#include "JSWebAssemblyCompileError.h"
#include "WebAssemblyCompileErrorPrototype.h"

namespace JSC {

const ClassInfo WebAssemblyCompileErrorConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyCompileErrorConstructor) };

static JSC_DECLARE_HOST_FUNCTION(constructJSWebAssemblyCompileError);
static JSC_DECLARE_HOST_FUNCTION(callJSWebAssemblyCompileError);

JSC_DEFINE_HOST_FUNCTION(constructJSWebAssemblyCompileError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue message = callFrame->argument(0);
    JSValue options = callFrame->argument(1);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, webAssemblyCompileErrorStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(ErrorInstance::create(globalObject, structure, message, options, nullptr, TypeNothing, ErrorType::Error, false)));
}

JSC_DEFINE_HOST_FUNCTION(callJSWebAssemblyCompileError, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    JSValue message = callFrame->argument(0);
    JSValue options = callFrame->argument(1);
    Structure* errorStructure = globalObject->webAssemblyCompileErrorStructure();
    return JSValue::encode(ErrorInstance::create(globalObject, errorStructure, message, options, nullptr, TypeNothing, ErrorType::Error, false));
}

WebAssemblyCompileErrorConstructor* WebAssemblyCompileErrorConstructor::create(VM& vm, Structure* structure, WebAssemblyCompileErrorPrototype* thisPrototype)
{
    auto* constructor = new (NotNull, allocateCell<WebAssemblyCompileErrorConstructor>(vm)) WebAssemblyCompileErrorConstructor(vm, structure);
    constructor->finishCreation(vm, thisPrototype);
    return constructor;
}

Structure* WebAssemblyCompileErrorConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

void WebAssemblyCompileErrorConstructor::finishCreation(VM& vm, WebAssemblyCompileErrorPrototype* prototype)
{
    Base::finishCreation(vm, 1, "CompileError"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum | PropertyAttribute::DontDelete);
}

WebAssemblyCompileErrorConstructor::WebAssemblyCompileErrorConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callJSWebAssemblyCompileError, constructJSWebAssemblyCompileError)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
