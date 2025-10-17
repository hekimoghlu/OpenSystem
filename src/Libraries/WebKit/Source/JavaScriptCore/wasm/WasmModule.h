/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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

#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#include "WasmCalleeGroup.h"
#include "WasmJS.h"
#include "WasmMemory.h"
#include "WasmOps.h"
#include <wtf/Expected.h>
#include <wtf/Lock.h>
#include <wtf/SharedTask.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace JSC {

class VM;
class JSWebAssemblyInstance;

namespace Wasm {

class LLIntPlan;
class IPIntPlan;
struct ModuleInformation;
enum class BindingFailure;

class Module : public ThreadSafeRefCounted<Module> {
public:
    using ValidationResult = Expected<Ref<Module>, String>;
    typedef void CallbackType(ValidationResult&&);
    using AsyncValidationCallback = RefPtr<SharedTask<CallbackType>>;

    static ValidationResult validateSync(VM&, Vector<uint8_t>&& source);
    static void validateAsync(VM&, Vector<uint8_t>&& source, Module::AsyncValidationCallback&&);

    static Ref<Module> create(LLIntPlan& plan)
    {
        return adoptRef(*new Module(plan));
    }
    static Ref<Module> create(IPIntPlan& plan)
    {
        return adoptRef(*new Module(plan));
    }

    Wasm::TypeIndex typeIndexFromFunctionIndexSpace(FunctionSpaceIndex functionIndexSpace) const;
    const Wasm::ModuleInformation& moduleInformation() const { return m_moduleInformation.get(); }

    Ref<CalleeGroup> compileSync(VM&, MemoryMode);
    void compileAsync(VM&, MemoryMode, CalleeGroup::AsyncCompilationCallback&&);

    JS_EXPORT_PRIVATE ~Module();

    CalleeGroup* calleeGroupFor(MemoryMode mode) { return m_calleeGroups[static_cast<uint8_t>(mode)].get(); }

    void copyInitialCalleeGroupToAllMemoryModes(MemoryMode);

    CodePtr<WasmEntryPtrTag> importFunctionStub(FunctionSpaceIndex importFunctionNum) { return m_wasmToJSExitStubs[importFunctionNum].code(); }

private:
    Ref<CalleeGroup> getOrCreateCalleeGroup(VM&, MemoryMode);

    Module(LLIntPlan&);
    Module(IPIntPlan&);
    Ref<ModuleInformation> m_moduleInformation;
    RefPtr<CalleeGroup> m_calleeGroups[numberOfMemoryModes];
    Ref<LLIntCallees> m_llintCallees;
    Ref<IPIntCallees> m_ipintCallees;
    FixedVector<MacroAssemblerCodeRef<WasmEntryPtrTag>> m_wasmToJSExitStubs;
    Lock m_lock;
};

} } // namespace JSC::Wasm

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(WEBASSEMBLY)
