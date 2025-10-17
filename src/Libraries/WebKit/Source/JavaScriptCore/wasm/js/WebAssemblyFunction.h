/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 14, 2025.
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
#pragma once

#if ENABLE(WEBASSEMBLY)

#include "ArityCheckMode.h"
#include "MacroAssemblerCodeRef.h"
#include "WasmCallee.h"
#include "WebAssemblyFunctionBase.h"
#include <wtf/Noncopyable.h>

namespace JSC {

class JSGlobalObject;
struct ProtoCallFrame;
class WebAssemblyInstance;

class WebAssemblyFunction final : public WebAssemblyFunctionBase {
public:
    using Base = WebAssemblyFunctionBase;

    static constexpr unsigned StructureFlags = Base::StructureFlags;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.webAssemblyFunctionSpace<mode>();
    }

    DECLARE_EXPORT_INFO;

    JS_EXPORT_PRIVATE static WebAssemblyFunction* create(VM&, JSGlobalObject*, Structure*, unsigned, const String&, JSWebAssemblyInstance*, Wasm::JSEntrypointCallee& jsEntrypoint, Wasm::Callee& wasmCallee, WasmToWasmImportableFunction::LoadLocation, Wasm::TypeIndex, RefPtr<const Wasm::RTT>&&);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    Wasm::JSEntrypointCallee* jsToWasmCallee() const { return m_boxedJSToWasmCallee.ptr(); }
    CodePtr<WasmEntryPtrTag> jsEntrypoint(ArityCheckMode arity)
    {
        ASSERT_UNUSED(arity, arity == ArityCheckNotRequired || arity == MustCheckArity);
        return m_boxedJSToWasmCallee->entrypoint();
    }

    CodePtr<JSEntryPtrTag> jsCallICEntrypoint()
    {
#if ENABLE(JIT)
        // Prep the entrypoint for the slow path.
        executable()->entrypointFor(CodeForCall, MustCheckArity);
        if (!m_jsToWasmICJITCode)
            m_jsToWasmICJITCode = signature().jsToWasmICEntrypoint();
        return m_jsToWasmICJITCode;
#else
        return nullptr;
#endif
    }

    static constexpr ptrdiff_t offsetOfBoxedWasmCallee() { return OBJECT_OFFSETOF(WebAssemblyFunction, m_boxedWasmCallee); }
    static constexpr ptrdiff_t offsetOfBoxedJSToWasmCallee() { return OBJECT_OFFSETOF(WebAssemblyFunction, m_boxedJSToWasmCallee); }
    static constexpr ptrdiff_t offsetOfFrameSize() { return OBJECT_OFFSETOF(WebAssemblyFunction, m_frameSize); }

private:
    DECLARE_VISIT_CHILDREN;
    WebAssemblyFunction(VM&, NativeExecutable*, JSGlobalObject*, Structure*, JSWebAssemblyInstance*, Wasm::JSEntrypointCallee& jsEntrypoint, Wasm::Callee& wasmCallee, WasmToWasmImportableFunction::LoadLocation entrypointLoadLocation, Wasm::TypeIndex, RefPtr<const Wasm::RTT>&&);

    CodePtr<JSEntryPtrTag> jsCallEntrypointSlow();

    // This is the callee needed by LLInt/IPInt
    // FIXME: Make this a Wasm::IPIntCallee once wasm LLInt goes away.
    Ref<Wasm::Callee, BoxedNativeCalleePtrTraits<Wasm::Callee>> m_boxedWasmCallee;
    // This let's the JS->Wasm interpreter find its metadata
    Ref<Wasm::JSEntrypointCallee, BoxedNativeCalleePtrTraits<Wasm::JSEntrypointCallee>> m_boxedJSToWasmCallee;
    uint32_t m_frameSize;
#if ENABLE(JIT)
    CodePtr<JSEntryPtrTag> m_jsToWasmICJITCode;
#endif
};

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
