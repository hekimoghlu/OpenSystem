/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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
#include "WebAssemblyGlobalPrototype.h"

#if ENABLE(WEBASSEMBLY)

#include "AuxiliaryBarrierInlines.h"
#include "GetterSetter.h"
#include "IntegrityInlines.h"
#include "JSCInlines.h"
#include "JSWebAssemblyGlobal.h"

namespace JSC {
static JSC_DECLARE_HOST_FUNCTION(webAssemblyGlobalProtoFuncValueOf);
static JSC_DECLARE_HOST_FUNCTION(webAssemblyGlobalProtoGetterFuncValue);
static JSC_DECLARE_HOST_FUNCTION(webAssemblyGlobalProtoSetterFuncValue);
static JSC_DECLARE_HOST_FUNCTION(webAssemblyGlobalProtoFuncType);
}

#include "WebAssemblyGlobalPrototype.lut.h"

namespace JSC {

const ClassInfo WebAssemblyGlobalPrototype::s_info = { "WebAssembly.Global"_s, &Base::s_info, &prototypeGlobalWebAssemblyGlobal, nullptr, CREATE_METHOD_TABLE(WebAssemblyGlobalPrototype) };

/* Source for WebAssemblyGlobalPrototype.lut.h
 @begin prototypeGlobalWebAssemblyGlobal
 valueOf webAssemblyGlobalProtoFuncValueOf Function 0
 type    webAssemblyGlobalProtoFuncType    Function 0

 @end
 */

static ALWAYS_INLINE JSWebAssemblyGlobal* getGlobal(JSGlobalObject* globalObject, VM& vm, JSValue v)
{
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    JSWebAssemblyGlobal* result = jsDynamicCast<JSWebAssemblyGlobal*>(v);
    if (!result) {
        throwException(globalObject, throwScope,
            createTypeError(globalObject, "expected |this| value to be an instance of WebAssembly.Global"_s));
        return nullptr;
    }
    Integrity::auditStructureID(result->structureID());
    return result;
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyGlobalProtoFuncValueOf, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyGlobal* global = getGlobal(globalObject, vm, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    RELEASE_AND_RETURN(throwScope, JSValue::encode(global->global()->get(globalObject)));
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyGlobalProtoFuncType, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyGlobal* global = getGlobal(globalObject, vm, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    JSObject* typeDescriptor = global->type(globalObject);
    if (!typeDescriptor)
        return throwVMTypeError(globalObject, throwScope, "WebAssembly.Global.prototype.type unable to produce type descriptor for the given global"_s);

    RELEASE_AND_RETURN(throwScope, JSValue::encode(typeDescriptor));
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyGlobalProtoGetterFuncValue, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSWebAssemblyGlobal* global = getGlobal(globalObject, vm, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
    RELEASE_AND_RETURN(throwScope, JSValue::encode(global->global()->get(globalObject)));
}

JSC_DEFINE_HOST_FUNCTION(webAssemblyGlobalProtoSetterFuncValue, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto throwScope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(callFrame->argumentCount() < 1))
        return JSValue::encode(throwException(globalObject, throwScope, createNotEnoughArgumentsError(globalObject)));

    JSWebAssemblyGlobal* global = getGlobal(globalObject, vm, callFrame->thisValue());
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());

    if (global->global()->mutability() == Wasm::Mutability::Immutable)
        return JSValue::encode(throwException(globalObject, throwScope, createTypeError(globalObject, "WebAssembly.Global.prototype.value attempts to modify immutable global value"_s)));

    global->global()->set(globalObject, callFrame->argument(0));
    RETURN_IF_EXCEPTION(throwScope, encodedJSValue());
    return JSValue::encode(jsUndefined());
}

WebAssemblyGlobalPrototype* WebAssemblyGlobalPrototype::create(VM& vm, JSGlobalObject* globalObject, Structure* structure)
{
    auto* object = new (NotNull, allocateCell<WebAssemblyGlobalPrototype>(vm)) WebAssemblyGlobalPrototype(vm, structure);
    object->finishCreation(vm, globalObject);
    return object;
}

Structure* WebAssemblyGlobalPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

void WebAssemblyGlobalPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));

    JSFunction* valueGetterFunction = JSFunction::create(vm, globalObject, 0, "get value"_s, webAssemblyGlobalProtoGetterFuncValue, ImplementationVisibility::Public);
    JSFunction* valueSetterFunction = JSFunction::create(vm, globalObject, 1, "set value"_s, webAssemblyGlobalProtoSetterFuncValue, ImplementationVisibility::Public);
    GetterSetter* valueAccessor = GetterSetter::create(vm, globalObject, valueGetterFunction, valueSetterFunction);
    putDirectNonIndexAccessorWithoutTransition(vm, Identifier::fromString(vm, "value"_s), valueAccessor, static_cast<unsigned>(PropertyAttribute::Accessor));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

WebAssemblyGlobalPrototype::WebAssemblyGlobalPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
