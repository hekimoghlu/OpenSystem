/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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
#include "WasmModule.h"

#if ENABLE(WEBASSEMBLY)

#include "WasmIPIntPlan.h"
#include "WasmLLIntPlan.h"
#include "WasmModuleInformation.h"
#include "WasmWorklist.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace Wasm {

Module::Module(LLIntPlan& plan)
    : m_moduleInformation(plan.takeModuleInformation())
    , m_llintCallees(LLIntCallees::createFromVector(plan.takeCallees()))
    , m_ipintCallees(IPIntCallees::create(0))
    , m_wasmToJSExitStubs(plan.takeWasmToJSExitStubs())
{
}

Module::Module(IPIntPlan& plan)
    : m_moduleInformation(plan.takeModuleInformation())
    , m_llintCallees(LLIntCallees::create(0))
    , m_ipintCallees(IPIntCallees::createFromVector(plan.takeCallees()))
    , m_wasmToJSExitStubs(plan.takeWasmToJSExitStubs())
{
}

Module::~Module() = default;

Wasm::TypeIndex Module::typeIndexFromFunctionIndexSpace(FunctionSpaceIndex functionIndexSpace) const
{
    return m_moduleInformation->typeIndexFromFunctionIndexSpace(functionIndexSpace);
}

static Module::ValidationResult makeValidationResult(LLIntPlan& plan)
{
    ASSERT(!plan.hasWork());
    if (plan.failed())
        return Unexpected<String>(plan.errorMessage());
    return Module::ValidationResult(Module::create(plan));
}

static Module::ValidationResult makeValidationResult(IPIntPlan& plan)
{
    ASSERT(!plan.hasWork());
    if (plan.failed())
        return Unexpected<String>(plan.errorMessage());
    return Module::ValidationResult(Module::create(plan));
}

static Plan::CompletionTask makeValidationCallback(Module::AsyncValidationCallback&& callback)
{
    return createSharedTask<Plan::CallbackType>([callback = WTFMove(callback)] (Plan& plan) {
        ASSERT(!plan.hasWork());
        if (Options::useWasmIPInt())
            callback->run(makeValidationResult(static_cast<IPIntPlan&>(plan)));
        else
            callback->run(makeValidationResult(static_cast<LLIntPlan&>(plan)));
    });
}

Module::ValidationResult Module::validateSync(VM& vm, Vector<uint8_t>&& source)
{
    if (Options::useWasmIPInt()) {
        Ref<IPIntPlan> plan = adoptRef(*new IPIntPlan(vm, WTFMove(source), CompilerMode::Validation, Plan::dontFinalize()));
        Wasm::ensureWorklist().enqueue(plan.get());
        plan->waitForCompletion();
        return makeValidationResult(plan.get());
    }
    Ref<LLIntPlan> plan = adoptRef(*new LLIntPlan(vm, WTFMove(source), CompilerMode::Validation, Plan::dontFinalize()));
    Wasm::ensureWorklist().enqueue(plan.get());
    plan->waitForCompletion();
    return makeValidationResult(plan.get());
}

void Module::validateAsync(VM& vm, Vector<uint8_t>&& source, Module::AsyncValidationCallback&& callback)
{
    if (Options::useWasmIPInt()) {
        Ref<Plan> plan = adoptRef(*new IPIntPlan(vm, WTFMove(source), CompilerMode::Validation, makeValidationCallback(WTFMove(callback))));
        Wasm::ensureWorklist().enqueue(WTFMove(plan));
    } else {
        Ref<Plan> plan = adoptRef(*new LLIntPlan(vm, WTFMove(source), CompilerMode::Validation, makeValidationCallback(WTFMove(callback))));
        Wasm::ensureWorklist().enqueue(WTFMove(plan));
    }
}

Ref<CalleeGroup> Module::getOrCreateCalleeGroup(VM& vm, MemoryMode mode)
{
    RefPtr<CalleeGroup> calleeGroup;
    Locker locker { m_lock };
    calleeGroup = m_calleeGroups[static_cast<uint8_t>(mode)];
    // If a previous attempt at a compile errored out, let's try again.
    // Compilations from valid modules can fail because OOM and cancellation.
    // It's worth retrying.
    // FIXME: We might want to back off retrying at some point:
    // https://bugs.webkit.org/show_bug.cgi?id=170607
    if (!calleeGroup || (calleeGroup->compilationFinished() && !calleeGroup->runnable())) {
        if (Options::useWasmIPInt())
            m_calleeGroups[static_cast<uint8_t>(mode)] = calleeGroup = CalleeGroup::createFromIPInt(vm, mode, const_cast<ModuleInformation&>(moduleInformation()), m_ipintCallees.copyRef());
        else
            m_calleeGroups[static_cast<uint8_t>(mode)] = calleeGroup = CalleeGroup::createFromLLInt(vm, mode, const_cast<ModuleInformation&>(moduleInformation()), m_llintCallees.copyRef());
    }
    return calleeGroup.releaseNonNull();
}

Ref<CalleeGroup> Module::compileSync(VM& vm, MemoryMode mode)
{
    Ref<CalleeGroup> calleeGroup = getOrCreateCalleeGroup(vm, mode);
    calleeGroup->waitUntilFinished();
    return calleeGroup;
}

void Module::compileAsync(VM& vm, MemoryMode mode, CalleeGroup::AsyncCompilationCallback&& task)
{
    Ref<CalleeGroup> calleeGroup = getOrCreateCalleeGroup(vm, mode);
    calleeGroup->compileAsync(vm, WTFMove(task));
}

void Module::copyInitialCalleeGroupToAllMemoryModes(MemoryMode initialMode)
{
    Locker locker { m_lock };
    ASSERT(m_calleeGroups[static_cast<uint8_t>(initialMode)]);
    const CalleeGroup& initialBlock = *m_calleeGroups[static_cast<uint8_t>(initialMode)];
    for (unsigned i = 0; i < numberOfMemoryModes; i++) {
        if (i == static_cast<uint8_t>(initialMode))
            continue;
        // We should only try to copy the group here if it hasn't already been created.
        // If it exists but is not runnable, it should get compiled during module evaluation.
        if (auto& group = m_calleeGroups[i]; !group)
            group = CalleeGroup::createFromExisting(static_cast<MemoryMode>(i), initialBlock);
    }
}

} } // namespace JSC::Wasm

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(WEBASSEMBLY)
