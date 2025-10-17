/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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

#include "WebAssemblyFunctionBase.h"

namespace JSC {

class WebAssemblyWrapperFunction final : public WebAssemblyFunctionBase {
public:
    using Base = WebAssemblyFunctionBase;

    static constexpr unsigned StructureFlags = Base::StructureFlags;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.webAssemblyWrapperFunctionSpace<mode>();
    }

    DECLARE_INFO;

    static WebAssemblyWrapperFunction* create(VM&, JSGlobalObject*, Structure*, JSObject*, unsigned importIndex, JSWebAssemblyInstance*, Wasm::TypeIndex, RefPtr<const Wasm::RTT>&&);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    JSObject* function() { return m_function.get(); }

private:
    WebAssemblyWrapperFunction(VM&, NativeExecutable*, JSGlobalObject*, Structure*, JSWebAssemblyInstance*, JSObject* function, Wasm::WasmOrJSImportableFunction&&, Wasm::WasmOrJSImportableFunctionCallLinkInfo*);
    DECLARE_VISIT_CHILDREN;

    WriteBarrier<JSObject> m_function;
};

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
