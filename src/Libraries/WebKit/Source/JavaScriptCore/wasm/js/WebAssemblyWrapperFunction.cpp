/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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
#include "WebAssemblyWrapperFunction.h"

#if ENABLE(WEBASSEMBLY)

#include "JSObjectInlines.h"
#include "JSWebAssemblyInstance.h"
#include "WasmTypeDefinitionInlines.h"

namespace JSC {

const ClassInfo WebAssemblyWrapperFunction::s_info = { "WebAssemblyWrapperFunction"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyWrapperFunction) };

static JSC_DECLARE_HOST_FUNCTION(callWebAssemblyWrapperFunction);
static JSC_DECLARE_HOST_FUNCTION(callWebAssemblyWrapperFunctionIncludingInvalidValues);

WebAssemblyWrapperFunction::WebAssemblyWrapperFunction(VM& vm, NativeExecutable* executable, JSGlobalObject* globalObject, Structure* structure, JSWebAssemblyInstance* instance, JSObject* function, Wasm::WasmOrJSImportableFunction&& importableFunction, WasmOrJSImportableFunctionCallLinkInfo* callLinkInfo)
    : Base(vm, executable, globalObject, structure, instance, WTFMove(importableFunction), callLinkInfo)
    , m_function(function, WriteBarrierEarlyInit)
{ }

WebAssemblyWrapperFunction* WebAssemblyWrapperFunction::create(VM& vm, JSGlobalObject* globalObject, Structure* structure, JSObject* function, unsigned importIndex, JSWebAssemblyInstance* instance, Wasm::TypeIndex typeIndex, RefPtr<const Wasm::RTT>&& rtt)
{
    ASSERT_WITH_MESSAGE(!function->inherits<WebAssemblyWrapperFunction>(), "We should never double wrap a wrapper function.");

    String name = emptyString();
    const auto& signature = Wasm::TypeInformation::getFunctionSignature(typeIndex);
    NativeExecutable* executable = nullptr;
    if (UNLIKELY(signature.argumentsOrResultsIncludeV128() || signature.argumentsOrResultsIncludeExnref()))
        executable = vm.getHostFunction(callWebAssemblyWrapperFunctionIncludingInvalidValues, ImplementationVisibility::Public, NoIntrinsic, callHostFunctionAsConstructor, nullptr, name);
    else
        executable = vm.getHostFunction(callWebAssemblyWrapperFunction, ImplementationVisibility::Public, NoIntrinsic, callHostFunctionAsConstructor, nullptr, name);

    RELEASE_ASSERT(JSValue(function).isCallable());
    WebAssemblyWrapperFunction* result = new (NotNull, allocateCell<WebAssemblyWrapperFunction>(vm)) WebAssemblyWrapperFunction(vm, executable, globalObject, structure, instance, function,
        Wasm::WasmOrJSImportableFunction {
            {
                {
                    &Wasm::NullWasmCallee,
                    { },
                    &instance->importFunctionInfo(importIndex)->importFunctionStub
                },
                typeIndex,
                rtt.get()
            },
            { },
            { }
        },
        instance->importFunctionInfo(importIndex));
    result->m_importableFunction.importFunction.set(vm, globalObject, function);
    result->finishCreation(vm, executable, signature.argumentCount(), name);
    return result;
}

Structure* WebAssemblyWrapperFunction::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    ASSERT(globalObject);
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSFunctionType, StructureFlags), info());
}

template<typename Visitor>
void WebAssemblyWrapperFunction::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    WebAssemblyWrapperFunction* thisObject = jsCast<WebAssemblyWrapperFunction*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);

    visitor.append(thisObject->m_function);
}

DEFINE_VISIT_CHILDREN(WebAssemblyWrapperFunction);

JSC_DEFINE_HOST_FUNCTION(callWebAssemblyWrapperFunction, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    WebAssemblyWrapperFunction* wasmFunction = jsCast<WebAssemblyWrapperFunction*>(callFrame->jsCallee());
    JSObject* function = wasmFunction->function();
    auto callData = JSC::getCallData(function);
    RELEASE_ASSERT(callData.type != CallData::Type::None);
    RELEASE_AND_RETURN(scope, JSValue::encode(call(globalObject, function, callData, jsUndefined(), ArgList(callFrame))));
}

JSC_DEFINE_HOST_FUNCTION(callWebAssemblyWrapperFunctionIncludingInvalidValues, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return throwVMTypeError(globalObject, scope, Wasm::errorMessageForExceptionType(Wasm::ExceptionType::TypeErrorInvalidValueUse));
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
