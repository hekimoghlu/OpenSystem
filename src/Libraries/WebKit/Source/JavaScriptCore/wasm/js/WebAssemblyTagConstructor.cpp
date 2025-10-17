/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
#include "WebAssemblyTagConstructor.h"

#if ENABLE(WEBASSEMBLY)

#include "Error.h"
#include "IteratorOperations.h"
#include "JSGlobalObject.h"
#include "JSWebAssemblyTag.h"
#include "WebAssemblyTagPrototype.h"

namespace JSC {

const ClassInfo WebAssemblyTagConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyTagConstructor) };

static JSC_DECLARE_HOST_FUNCTION(callJSWebAssemblyTag);
static JSC_DECLARE_HOST_FUNCTION(constructJSWebAssemblyTag);

JSC_DEFINE_HOST_FUNCTION(constructJSWebAssemblyTag, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    
    if (callFrame->argumentCount() < 1)
        return throwVMTypeError(globalObject, scope, "WebAssembly.Tag constructor expects the tag type as the first argument."_s);

    JSValue tagTypeValue = callFrame->argument(0);
    JSValue signatureObject = tagTypeValue.get(globalObject, Identifier::fromString(vm, "parameters"_s));
    RETURN_IF_EXCEPTION(scope, { });

    if (!signatureObject.isObject())
        return throwVMTypeError(globalObject, scope, "WebAssembly.Tag constructor expects a tag type with the 'parameters' property."_s);

    Vector<Wasm::Type, 16> parameters;
    forEachInIterable(globalObject, signatureObject, [&] (auto& vm, auto* globalObject, JSValue nextType) -> void {
        auto scope = DECLARE_THROW_SCOPE(vm);

        Wasm::Type type;
        String valueString = nextType.toWTFString(globalObject);
        RETURN_IF_EXCEPTION(scope, void());
        if (valueString == "i32"_s)
            type = Wasm::Types::I32;
        else if (valueString == "i64"_s)
            type = Wasm::Types::I64;
        else if (valueString == "f32"_s)
            type = Wasm::Types::F32;
        else if (valueString == "f64"_s)
            type = Wasm::Types::F64;
        else if (valueString == "v128"_s)
            type = Wasm::Types::V128;
        else if (valueString == "funcref"_s || valueString == "anyfunc"_s)
            type = Wasm::funcrefType();
        else if (valueString == "externref"_s)
            type = Wasm::externrefType();
        else {
            throwTypeError(globalObject, scope, "WebAssembly.Tag constructor expects the 'parameters' field of the first argument to be a sequence of WebAssembly value types."_s);
            return;
        }

        parameters.append(type);
    });
    RETURN_IF_EXCEPTION(scope, { });

    RefPtr<Wasm::TypeDefinition> typeDefinition = Wasm::TypeInformation::typeDefinitionForFunction({ }, parameters);
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, webAssemblyTagStructure, asObject(callFrame->newTarget()), callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });
    RELEASE_AND_RETURN(scope, JSValue::encode(JSWebAssemblyTag::create(vm, globalObject, structure, Wasm::Tag::create(typeDefinition.releaseNonNull()).get())));
}

JSC_DEFINE_HOST_FUNCTION(callJSWebAssemblyTag, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "WebAssembly.Tag"_s));
}

WebAssemblyTagConstructor* WebAssemblyTagConstructor::create(VM& vm, Structure* structure, WebAssemblyTagPrototype* thisPrototype)
{
    auto* constructor = new (NotNull, allocateCell<WebAssemblyTagConstructor>(vm)) WebAssemblyTagConstructor(vm, structure);
    constructor->finishCreation(vm, thisPrototype);
    return constructor;
}

Structure* WebAssemblyTagConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

void WebAssemblyTagConstructor::finishCreation(VM& vm, WebAssemblyTagPrototype* prototype)
{
    Base::finishCreation(vm, 1, "Tag"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

WebAssemblyTagConstructor::WebAssemblyTagConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callJSWebAssemblyTag, constructJSWebAssemblyTag)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
