/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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

#if ENABLE(WEBASSEMBLY_BBQJIT)

#include "CompilationResult.h"
#include "WasmCallee.h"
#include "WasmEntryPlan.h"
#include "WasmModuleInformation.h"
#include "tools/FunctionAllowlist.h"
#include <wtf/Bag.h>
#include <wtf/Function.h>
#include <wtf/SharedTask.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Vector.h>

namespace JSC {

class CallLinkInfo;

namespace Wasm {

class BBQCallee;
class CalleeGroup;
class JSEntrypointCallee;

class BBQPlan final : public Plan {
public:
    using Base = Plan;

    static Ref<BBQPlan> create(VM& vm, Ref<ModuleInformation>&& info, FunctionCodeIndex functionIndex, std::optional<bool> hasExceptionHandlers, Ref<CalleeGroup>&& calleeGroup, CompletionTask&& completionTask)
    {
        return adoptRef(*new BBQPlan(vm, WTFMove(info), functionIndex, hasExceptionHandlers, WTFMove(calleeGroup), WTFMove(completionTask)));
    }

    bool hasWork() const final { return !m_completed; }
    void work(CompilationEffort) final;
    bool multiThreaded() const final { return false; }

    static FunctionAllowlist& ensureGlobalBBQAllowlist();


private:
    BBQPlan(VM&, Ref<ModuleInformation>&&, FunctionCodeIndex functionIndex, std::optional<bool> hasExceptionHandlers, Ref<CalleeGroup>&&, CompletionTask&&);

    bool dumpDisassembly(CompilationContext&, LinkBuffer&, FunctionCodeIndex functionIndex, const TypeDefinition&, FunctionSpaceIndex functionIndexSpace);

    std::unique_ptr<InternalFunction> compileFunction(FunctionCodeIndex functionIndex, BBQCallee&, CompilationContext&, Vector<UnlinkedWasmToWasmCall>&);
    bool isComplete() const final { return m_completed; }
    void complete() WTF_REQUIRES_LOCK(m_lock) final
    {
        m_completed = true;
        runCompletionTasks();
    }

    Ref<CalleeGroup> m_calleeGroup;
    FunctionCodeIndex m_functionIndex;
    bool m_completed { false };
    std::optional<bool> m_hasExceptionHandlers;
};


} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY_BBQJIT)
