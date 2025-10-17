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
#pragma once

#if ENABLE(WEBASSEMBLY)

#include "JSFunction.h"
#include "WasmFormat.h"

namespace JSC {

class JSGlobalObject;
class JSWebAssemblyInstance;
using Wasm::WasmToWasmImportableFunction;
using Wasm::WasmOrJSImportableFunctionCallLinkInfo;

class WebAssemblyFunctionBase : public JSFunction {
public:
    using Base = JSFunction;

    static constexpr unsigned StructureFlags = Base::StructureFlags;

    DECLARE_INFO;

    JSWebAssemblyInstance* instance() const { return m_instance.get(); }

    Wasm::TypeIndex typeIndex() const { return m_importableFunction.typeIndex; }
    Wasm::Type type() const { return { Wasm::TypeKind::Ref, typeIndex() }; }
    WasmToWasmImportableFunction::LoadLocation entrypointLoadLocation() const { return m_importableFunction.entrypointLoadLocation; }
    const uintptr_t* boxedWasmCalleeLoadLocation() const { return m_importableFunction.boxedWasmCalleeLoadLocation; }
    const Wasm::WasmOrJSImportableFunction& importableFunction() const { return m_importableFunction; }
    const Wasm::RTT* rtt() const { return m_importableFunction.rtt; }
    const Wasm::FunctionSignature& signature() const;
    WasmOrJSImportableFunctionCallLinkInfo* callLinkInfo() const { return m_callLinkInfo; }

    static constexpr ptrdiff_t offsetOfInstance() { return OBJECT_OFFSETOF(WebAssemblyFunctionBase, m_instance); }

    static constexpr ptrdiff_t offsetOfSignatureIndex() { return OBJECT_OFFSETOF(WebAssemblyFunctionBase, m_importableFunction) + WasmToWasmImportableFunction::offsetOfSignatureIndex(); }

    static constexpr ptrdiff_t offsetOfEntrypointLoadLocation() { return OBJECT_OFFSETOF(WebAssemblyFunctionBase, m_importableFunction) + WasmToWasmImportableFunction::offsetOfEntrypointLoadLocation(); }
    static constexpr ptrdiff_t offsetOfBoxedWasmCalleeLoadLocation() { return OBJECT_OFFSETOF(WebAssemblyFunctionBase, m_importableFunction) + WasmToWasmImportableFunction::offsetOfBoxedWasmCalleeLoadLocation(); }

    static constexpr ptrdiff_t offsetOfRTT() { return OBJECT_OFFSETOF(WebAssemblyFunctionBase, m_importableFunction) + WasmToWasmImportableFunction::offsetOfRTT(); }

protected:
    DECLARE_VISIT_CHILDREN;
    void finishCreation(VM&, NativeExecutable*, unsigned length, const String& name);
    WebAssemblyFunctionBase(VM&, NativeExecutable*, JSGlobalObject*, Structure*, JSWebAssemblyInstance*, Wasm::WasmOrJSImportableFunction&&, Wasm::WasmOrJSImportableFunctionCallLinkInfo*);

    Wasm::WasmOrJSImportableFunction m_importableFunction;
    // It's safe to just hold the raw WasmToWasmImportableFunctionCallLinkInfo because we have a reference
    // to our Instance, which points to the CodeBlock, which points to the Module
    // that exported us, which ensures that the actual Signature/RTT/code doesn't get deallocated.
    Wasm::WasmOrJSImportableFunctionCallLinkInfo* m_callLinkInfo;
    WriteBarrier<JSWebAssemblyInstance> m_instance;
};

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
