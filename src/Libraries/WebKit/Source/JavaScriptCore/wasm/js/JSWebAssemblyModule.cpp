/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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
#include "JSWebAssemblyModule.h"

#if ENABLE(WEBASSEMBLY)

#include "JSCInlines.h"
#include "JSWebAssemblyCompileError.h"
#include "JSWebAssemblyLinkError.h"
#include "WasmBinding.h"
#include "WasmFormat.h"
#include "WasmModule.h"
#include "WasmModuleInformation.h"
#include <wtf/StdLibExtras.h>
#include <wtf/text/MakeString.h>

namespace JSC {

const ClassInfo JSWebAssemblyModule::s_info = { "WebAssembly.Module"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWebAssemblyModule) };

JSWebAssemblyModule* JSWebAssemblyModule::create(VM& vm, Structure* structure, Ref<Wasm::Module>&& result)
{
    auto* module = new (NotNull, allocateCell<JSWebAssemblyModule>(vm)) JSWebAssemblyModule(vm, structure, WTFMove(result));
    module->finishCreation(vm);
    return module;
}

Structure* JSWebAssemblyModule::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(WebAssemblyModuleType, StructureFlags), info());
}


JSWebAssemblyModule::JSWebAssemblyModule(VM& vm, Structure* structure, Ref<Wasm::Module>&& module)
    : Base(vm, structure)
    , m_module(WTFMove(module))
{
}

void JSWebAssemblyModule::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));

    // On success, a new WebAssembly.Module object is returned with [[Module]] set to the validated Ast.module.
    SymbolTable* exportSymbolTable = SymbolTable::create(vm);
    const Wasm::ModuleInformation& moduleInformation = m_module->moduleInformation();
    {
        auto offset = exportSymbolTable->takeNextScopeOffset(NoLockingNecessary);
        exportSymbolTable->set(NoLockingNecessary, vm.propertyNames->starNamespacePrivateName.impl(), SymbolTableEntry(VarOffset(offset)));
    }
    for (auto& exp : moduleInformation.exports) {
        auto offset = exportSymbolTable->takeNextScopeOffset(NoLockingNecessary);
        exportSymbolTable->set(NoLockingNecessary, makeAtomString(exp.field).impl(), SymbolTableEntry(VarOffset(offset)));
    }

    m_exportSymbolTable.set(vm, this, exportSymbolTable);
}

void JSWebAssemblyModule::destroy(JSCell* cell)
{
    static_cast<JSWebAssemblyModule*>(cell)->JSWebAssemblyModule::~JSWebAssemblyModule();
    Wasm::TypeInformation::tryCleanup();
}

const Wasm::ModuleInformation& JSWebAssemblyModule::moduleInformation() const
{
    return m_module->moduleInformation();
}

SymbolTable* JSWebAssemblyModule::exportSymbolTable() const
{
    return m_exportSymbolTable.get();
}

Wasm::TypeIndex JSWebAssemblyModule::typeIndexFromFunctionIndexSpace(Wasm::FunctionSpaceIndex functionIndexSpace) const
{
    return m_module->typeIndexFromFunctionIndexSpace(functionIndexSpace);
}

Wasm::Module& JSWebAssemblyModule::module()
{
    return m_module.get();
}

template<typename Visitor>
void JSWebAssemblyModule::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    JSWebAssemblyModule* thisObject = jsCast<JSWebAssemblyModule*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());

    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_exportSymbolTable);
}

DEFINE_VISIT_CHILDREN(JSWebAssemblyModule);

} // namespace JSC

#endif // ENABLE(WEBASSEMBLY)
