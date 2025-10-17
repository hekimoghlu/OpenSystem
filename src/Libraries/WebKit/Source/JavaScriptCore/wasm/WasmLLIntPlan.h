/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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

#include "WasmCallee.h"
#include "WasmEntryPlan.h"
#include "WasmFunctionCodeBlockGenerator.h"

namespace JSC {

class CallLinkInfo;

namespace Wasm {

class LLIntCallee;
class JSEntrypointCallee;
class StreamingCompiler;

using JSEntrypointCalleeMap = UncheckedKeyHashMap<uint32_t, RefPtr<JSEntrypointCallee>, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>>;

using TailCallGraph = UncheckedKeyHashMap<uint32_t, UncheckedKeyHashSet<uint32_t, IntHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>>, IntHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>>;

class LLIntPlan final : public EntryPlan {
    using Base = EntryPlan;

public:
    JS_EXPORT_PRIVATE LLIntPlan(VM&, Vector<uint8_t>&&, CompilerMode, CompletionTask&&);
    LLIntPlan(VM&, Ref<ModuleInformation>, const Ref<LLIntCallee>*, CompletionTask&&);
    LLIntPlan(VM&, Ref<ModuleInformation>, CompilerMode, CompletionTask&&); // For StreamingCompiler.

    Vector<Ref<LLIntCallee>>&& takeCallees()
    {
        RELEASE_ASSERT(!failed() && !hasWork());
        return WTFMove(m_calleesVector);
    }

    JSEntrypointCalleeMap&& takeJSCallees()
    {
        RELEASE_ASSERT(!failed() && !hasWork());
        return WTFMove(m_jsEntrypointCallees);
    }

    bool hasWork() const final
    {
        return m_state < State::Compiled;
    }

    void work(CompilationEffort) final;

    bool didReceiveFunctionData(FunctionCodeIndex, const FunctionData&) final;

    void compileFunction(FunctionCodeIndex functionIndex) final;

    void completeInStreaming();
    void didCompileFunctionInStreaming();
    void didFailInStreaming(String&&);

private:
    bool prepareImpl() final;
    void didCompleteCompilation() WTF_REQUIRES_LOCK(m_lock) final;

    void addTailCallEdge(uint32_t, uint32_t);
    void computeTransitiveTailCalls() const;

    bool ensureEntrypoint(LLIntCallee&, FunctionCodeIndex functionIndex);

    Vector<std::unique_ptr<FunctionCodeBlockGenerator>> m_wasmInternalFunctions;
    const Ref<LLIntCallee>* m_callees { nullptr };
    Vector<Ref<LLIntCallee>> m_calleesVector;
    Vector<RefPtr<JSEntrypointCallee>> m_entrypoints;
    JSEntrypointCalleeMap m_jsEntrypointCallees;
    TailCallGraph m_tailCallGraph;
};


} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
