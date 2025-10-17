/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#include "WebAssemblyFunction.h"

#if ENABLE(WEBASSEMBLY)

#include "JITOpaqueByproducts.h"
#include "JSCJSValueInlines.h"
#include "JSObject.h"
#include "JSObjectInlines.h"
#include "JSToWasm.h"
#include "JSWebAssemblyHelpers.h"
#include "JSWebAssemblyInstance.h"
#include "JSWebAssemblyMemory.h"
#include "JSWebAssemblyRuntimeError.h"
#include "LLIntThunks.h"
#include "LinkBuffer.h"
#include "NativeExecutable.h"
#include "ProtoCallFrameInlines.h"
#include "SlotVisitorInlines.h"
#include "StructureInlines.h"
#include "WasmCallee.h"
#include "WasmCallingConvention.h"
#include "WasmContext.h"
#include "WasmFormat.h"
#include "WasmMemory.h"
#include "WasmMemoryInformation.h"
#include "WasmModuleInformation.h"
#include "WasmOperations.h"
#include "WasmTypeDefinitionInlines.h"
#include <wtf/StackPointer.h>
#include <wtf/SystemTracing.h>

namespace JSC {

const ClassInfo WebAssemblyFunction::s_info = { "WebAssemblyFunction"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WebAssemblyFunction) };

static JSC_DECLARE_HOST_FUNCTION(callWebAssemblyFunction);

JSC_DEFINE_HOST_FUNCTION(callWebAssemblyFunction, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    WebAssemblyFunction* wasmFunction = jsCast<WebAssemblyFunction*>(callFrame->jsCallee());

    // Note: we specifically use the WebAssemblyFunction as the callee to begin with in the ProtoCallFrame.
    // The reason for this is that calling into the llint may stack overflow, and the stack overflow
    // handler might read the global object from the callee.
    // For wasm, we setup |codeBlock| and |this| differently.
    //     |codeBlock| : JS entry wrapper expects a JSWebAssemblyInstance* as the |codeBlock| value argument.
    ProtoCallFrame protoCallFrame;
    protoCallFrame.init(nullptr, globalObject, wasmFunction, JSValue(), callFrame->argumentCountIncludingThis(), std::bit_cast<EncodedJSValue*>(callFrame->addressOfArgumentsStart()));
    protoCallFrame.setWasmInstance(wasmFunction->instance());

    return vmEntryToWasm(wasmFunction->jsEntrypoint(MustCheckArity).taggedPtr(), &vm, &protoCallFrame);
}

WebAssemblyFunction* WebAssemblyFunction::create(VM& vm, JSGlobalObject* globalObject, Structure* structure, unsigned length, const String& name, JSWebAssemblyInstance* instance, Wasm::JSEntrypointCallee& jsEntrypoint, Wasm::Callee& wasmCallee, Wasm::WasmToWasmImportableFunction::LoadLocation wasmToWasmEntrypointLoadLocation, Wasm::TypeIndex typeIndex, RefPtr<const Wasm::RTT>&& rtt)
{
    NativeExecutable* base = vm.getHostFunction(callWebAssemblyFunction, ImplementationVisibility::Public, WasmFunctionIntrinsic, callHostFunctionAsConstructor, nullptr, String());
    // Since ClosureCall uses this executable as an identity for Wasm CallIC thunk, we need to make it diversified.
    NativeExecutable* executable = NativeExecutable::create(vm, base->generatedJITCodeForCall(), callWebAssemblyFunction, base->generatedJITCodeForConstruct(), callHostFunctionAsConstructor, ImplementationVisibility::Public, name);
    WebAssemblyFunction* function = new (NotNull, allocateCell<WebAssemblyFunction>(vm)) WebAssemblyFunction(vm, executable, globalObject, structure, instance, jsEntrypoint, wasmCallee, wasmToWasmEntrypointLoadLocation, typeIndex, WTFMove(rtt));
    function->finishCreation(vm, executable, length, name);
    return function;
}

Structure* WebAssemblyFunction::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    ASSERT(globalObject);
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSFunctionType, StructureFlags), info());
}

WebAssemblyFunction::WebAssemblyFunction(VM& vm, NativeExecutable* executable, JSGlobalObject* globalObject, Structure* structure, JSWebAssemblyInstance* instance, Wasm::JSEntrypointCallee& jsEntrypoint, Wasm::Callee& wasmCallee, Wasm::WasmToWasmImportableFunction::LoadLocation wasmToWasmEntrypointLoadLocation, Wasm::TypeIndex typeIndex, RefPtr<const Wasm::RTT>&& rtt)
    : Base { vm, executable, globalObject, structure, instance, Wasm::WasmOrJSImportableFunction { { { reinterpret_cast<uintptr_t*>(&m_boxedWasmCallee), { vm, globalObject, instance }, wasmToWasmEntrypointLoadLocation }, typeIndex, rtt.get() }, { }, { } }, nullptr }
    , m_boxedWasmCallee(wasmCallee)
    , m_boxedJSToWasmCallee(jsEntrypoint)
    , m_frameSize(jsEntrypoint.frameSize())
{ }

template<typename Visitor>
void WebAssemblyFunction::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    WebAssemblyFunction* thisObject = jsCast<WebAssemblyFunction*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());

    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(WebAssemblyFunction);

void WebAssemblyFunction::destroy(JSCell* cell)
{
    static_cast<WebAssemblyFunction*>(cell)->WebAssemblyFunction::~WebAssemblyFunction();
}

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
