/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 7, 2022.
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
#include "WebAssemblyExceptionConstructor.h"

#if ENABLE(WEBASSEMBLY)

#include "IteratorOperations.h"
#include "JITOpaqueByproducts.h"
#include "JSCInlines.h"
#include "JSWebAssemblyException.h"
#include "JSWebAssemblyHelpers.h"
#include "JSWebAssemblyTag.h"
#include "WebAssemblyExceptionPrototype.h"

namespace JSC {

const ClassInfo WebAssemblyExceptionConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyExceptionConstructor) };

static JSC_DECLARE_HOST_FUNCTION(constructJSWebAssemblyException);
static JSC_DECLARE_HOST_FUNCTION(callJSWebAssemblyException);

JSC_DEFINE_HOST_FUNCTION(constructJSWebAssemblyException, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    auto& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue tagValue = callFrame->argument(0);
    JSValue tagParameters = callFrame->argument(1);

    auto tag = jsDynamicCast<JSWebAssemblyTag*>(tagValue);
    if (!tag)
        return throwVMTypeError(globalObject, scope, "WebAssembly.Exception constructor expects the first argument to be a WebAssembly.Tag"_s);

    if (UNLIKELY(&tag->tag() == &Wasm::Tag::jsExceptionTag()))
        return throwVMTypeError(globalObject, scope, "WebAssembly.Exception constructor does not accept WebAssembly.JSTag"_s);

    const auto& tagFunctionType = tag->type();
    MarkedArgumentBuffer values;
    values.ensureCapacity(tagFunctionType.argumentCount());
    forEachInIterable(globalObject, tagParameters, [&] (VM&, JSGlobalObject*, JSValue nextValue) {
        values.append(nextValue);
        if (UNLIKELY(values.hasOverflowed()))
            throwOutOfMemoryError(globalObject, scope);
    });
    RETURN_IF_EXCEPTION(scope, { });

    if (values.size() != tagFunctionType.argumentCount())
        return throwVMTypeError(globalObject, scope, "WebAssembly.Exception constructor expects the number of paremeters in WebAssembly.Tag to match the tags parameter count."_s);

    // Any GC'd values in here will be marked by the MarkedArugementBuffer until stored in the Exception.
    FixedVector<uint64_t> payload(values.size());
    for (unsigned i = 0; i < values.size(); ++i) {
        auto type = tagFunctionType.argumentType(i);
        if (UNLIKELY(type.kind == Wasm::TypeKind::V128 || isExnref(type)))
            return throwVMTypeError(globalObject, scope, "WebAssembly.Exception constructor expects payload includes neither v128 nor exnref."_s);
        payload[i] = toWebAssemblyValue(globalObject, type, values.at(i));
        RETURN_IF_EXCEPTION(scope, { });
    }

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* structure = JSC_GET_DERIVED_STRUCTURE(vm, webAssemblyExceptionStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(JSWebAssemblyException::create(vm, structure, tag->tag(), WTFMove(payload))));
}

JSC_DEFINE_HOST_FUNCTION(callJSWebAssemblyException, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "WebAssembly.Exception"_s));
}

WebAssemblyExceptionConstructor* WebAssemblyExceptionConstructor::create(VM& vm, Structure* structure, WebAssemblyExceptionPrototype* thisPrototype)
{
    auto* constructor = new (NotNull, allocateCell<WebAssemblyExceptionConstructor>(vm)) WebAssemblyExceptionConstructor(vm, structure);
    constructor->finishCreation(vm, thisPrototype);
    return constructor;
}

Structure* WebAssemblyExceptionConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

void WebAssemblyExceptionConstructor::finishCreation(VM& vm, WebAssemblyExceptionPrototype* prototype)
{
    Base::finishCreation(vm, 1, "Exception"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum | PropertyAttribute::DontDelete);
}

WebAssemblyExceptionConstructor::WebAssemblyExceptionConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callJSWebAssemblyException, constructJSWebAssemblyException)
{
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
