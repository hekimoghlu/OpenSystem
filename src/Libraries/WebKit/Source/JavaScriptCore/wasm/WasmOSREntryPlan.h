/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 22, 2024.
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

#if ENABLE(WEBASSEMBLY_OMGJIT)

#include "WasmCallee.h"
#include "WasmContext.h"
#include "WasmModule.h"
#include "WasmOperations.h"
#include "WasmPlan.h"

namespace JSC {

class CallLinkInfo;

namespace Wasm {

class OSREntryPlan final : public Plan {
public:
    using Base = Plan;

    bool hasWork() const final { return !m_completed; }
    void work(CompilationEffort) final;
    bool multiThreaded() const final { return false; }

    // Note: CompletionTask should not hold a reference to the Plan otherwise there will be a reference cycle.
    OSREntryPlan(VM&, Ref<Module>&&, Ref<Callee>&&, FunctionCodeIndex functionIndex, std::optional<bool> hasExceptionHandlers, uint32_t loopIndex, MemoryMode, CompletionTask&&);

private:
    // For some reason friendship doesn't extend to parent classes...
    using Base::m_lock;

    void dumpDisassembly(CompilationContext&, LinkBuffer&, FunctionCodeIndex functionIndex, const TypeDefinition&, FunctionSpaceIndex functionIndexSpace);
    bool isComplete() const final { return m_completed; }
    void complete() WTF_REQUIRES_LOCK(m_lock) final
    {
        m_completed = true;
        runCompletionTasks();
    }

    Ref<Module> m_module;
    Ref<CalleeGroup> m_calleeGroup;
    Ref<Callee> m_callee;
    bool m_completed { false };
    std::optional<bool> m_hasExceptionHandlers;
    FunctionCodeIndex m_functionIndex;
    uint32_t m_loopIndex;
};

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY_OMGJIT)
