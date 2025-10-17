/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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

#include "CallLinkInfo.h"
#include "JSDestructibleObject.h"
#include "JSObject.h"
#include "MemoryMode.h"
#include "WasmFormat.h"
#include <wtf/Bag.h>
#include <wtf/Expected.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/text/WTFString.h>

namespace JSC {

namespace Wasm {
class Module;
struct ModuleInformation;
class Plan;
}

class JSWebAssemblyMemory;
class SymbolTable;

class JSWebAssemblyModule final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.webAssemblyModuleSpace<mode>();
    }

    DECLARE_EXPORT_INFO;

    JS_EXPORT_PRIVATE static JSWebAssemblyModule* create(VM&, Structure*, Ref<Wasm::Module>&&);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    const Wasm::ModuleInformation& moduleInformation() const;
    SymbolTable* exportSymbolTable() const;
    Wasm::TypeIndex typeIndexFromFunctionIndexSpace(Wasm::FunctionSpaceIndex functionIndexSpace) const;

    JS_EXPORT_PRIVATE Wasm::Module& module();

private:
    JSWebAssemblyModule(VM&, Structure*, Ref<Wasm::Module>&&);
    void finishCreation(VM&);
    DECLARE_VISIT_CHILDREN;

    Ref<Wasm::Module> m_module;
    WriteBarrier<SymbolTable> m_exportSymbolTable;
};

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
