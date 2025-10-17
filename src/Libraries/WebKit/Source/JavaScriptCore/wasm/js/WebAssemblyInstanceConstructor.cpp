/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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
#include "WebAssemblyInstanceConstructor.h"

#if ENABLE(WEBASSEMBLY)

#include "JSCInlines.h"
#include "JSWebAssemblyInstance.h"
#include "JSWebAssemblyModule.h"
#include "WebAssemblyInstancePrototype.h"

namespace JSC {

const ClassInfo WebAssemblyInstanceConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyInstanceConstructor) };

static JSC_DECLARE_HOST_FUNCTION(constructJSWebAssemblyInstance);
static JSC_DECLARE_HOST_FUNCTION(callJSWebAssemblyInstance);

using Wasm::Plan;

JSC_DEFINE_HOST_FUNCTION(constructJSWebAssemblyInstance, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    // If moduleObject is not a WebAssembly.Module instance, a TypeError is thrown.
    JSWebAssemblyModule* module = jsDynamicCast<JSWebAssemblyModule*>(callFrame->argument(0));
    if (!module)
        return JSValue::encode(throwException(globalObject, scope, createTypeError(globalObject, "first argument to WebAssembly.Instance must be a WebAssembly.Module"_s, defaultSourceAppender, runtimeTypeForValue(callFrame->argument(0)))));

    // If the importObject parameter is not undefined and Type(importObject) is not Object, a TypeError is thrown.
    JSValue importArgument = callFrame->argument(1);
    JSObject* importObject = importArgument.getObject();
    if (!importArgument.isUndefined() && !importObject)
        return JSValue::encode(throwException(globalObject, scope, createTypeError(globalObject, "second argument to WebAssembly.Instance must be undefined or an Object"_s, defaultSourceAppender, runtimeTypeForValue(importArgument))));

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* instanceStructure = JSC_GET_DERIVED_STRUCTURE(vm, webAssemblyInstanceStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    JSWebAssemblyInstance* instance = JSWebAssemblyInstance::tryCreate(vm, instanceStructure, globalObject, JSWebAssemblyInstance::createPrivateModuleKey(), module, importObject, Wasm::CreationMode::FromJS);
    RETURN_IF_EXCEPTION(scope, { });

    instance->initializeImports(globalObject, importObject, Wasm::CreationMode::FromJS);
    RETURN_IF_EXCEPTION(scope, { });

    instance->finalizeCreation(vm, globalObject, module->module().compileSync(vm, instance->memoryMode()), Wasm::CreationMode::FromJS);
    RETURN_IF_EXCEPTION(scope, { });
    return JSValue::encode(instance);
}

JSC_DEFINE_HOST_FUNCTION(callJSWebAssemblyInstance, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "WebAssembly.Instance"_s));
}

WebAssemblyInstanceConstructor* WebAssemblyInstanceConstructor::create(VM& vm, Structure* structure, WebAssemblyInstancePrototype* thisPrototype)
{
    auto* constructor = new (NotNull, allocateCell<WebAssemblyInstanceConstructor>(vm)) WebAssemblyInstanceConstructor(vm, structure);
    constructor->finishCreation(vm, thisPrototype);
    return constructor;
}

Structure* WebAssemblyInstanceConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

void WebAssemblyInstanceConstructor::finishCreation(VM& vm, WebAssemblyInstancePrototype* prototype)
{
    Base::finishCreation(vm, 1, "Instance"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

WebAssemblyInstanceConstructor::WebAssemblyInstanceConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callJSWebAssemblyInstance, constructJSWebAssemblyInstance)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)

