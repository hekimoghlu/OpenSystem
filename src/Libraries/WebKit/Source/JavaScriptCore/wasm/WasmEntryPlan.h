/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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

#include "WasmModuleInformation.h"
#include "WasmPlan.h"
#include "WasmStreamingParser.h"
#include <wtf/Function.h>
#include <wtf/SharedTask.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/MakeString.h>

namespace JSC {

class CallLinkInfo;

namespace Wasm {

enum class BindingFailure;

class EntryPlan : public Plan, public StreamingParserClient {
public:
    using Base = Plan;

    // Note: CompletionTask should not hold a reference to the Plan otherwise there will be a reference cycle.
    EntryPlan(VM&, Ref<ModuleInformation>, CompilerMode, CompletionTask&&);
    JS_EXPORT_PRIVATE EntryPlan(VM&, Vector<uint8_t>&&, CompilerMode, CompletionTask&&);

    ~EntryPlan() override = default;

    void prepare();

    void compileFunctions(CompilationEffort);

    Ref<ModuleInformation>&& takeModuleInformation()
    {
        RELEASE_ASSERT(!failed() && !hasWork());
        return WTFMove(m_moduleInformation);
    }

    Vector<MacroAssemblerCodeRef<WasmEntryPtrTag>>&& takeWasmToWasmExitStubs()
    {
        RELEASE_ASSERT(!failed() && !hasWork());
        return WTFMove(m_wasmToWasmExitStubs);
    }

    Vector<Vector<UnlinkedWasmToWasmCall>> takeWasmToWasmCallsites()
    {
        RELEASE_ASSERT(!failed() && !hasWork());
        return WTFMove(m_unlinkedWasmToWasmCalls);
    }

    Vector<MacroAssemblerCodeRef<WasmEntryPtrTag>> takeWasmToJSExitStubs()
    {
        RELEASE_ASSERT(!failed() && !hasWork());
        return WTFMove(m_wasmToJSExitStubs);
    }

    enum class State : uint8_t {
        Initial,
        Validated,
        Prepared,
        Compiled,
        Completed // We should only move to Completed if we are holding the lock.
    };

    // FIXME: This seems like it should be `m_state == State::Prepared`?
    bool multiThreaded() const override { return m_state >= State::Prepared; }

    bool completeSyncIfPossible();

    virtual void completeInStreaming() = 0;
    virtual void didCompileFunctionInStreaming() = 0;
    virtual void didFailInStreaming(String&&) = 0;

private:
    class ThreadCountHolder;
    friend class ThreadCountHolder;
    friend class StreamingPlan;

protected:
    // For some reason friendship doesn't extend to parent classes...
    using Base::m_lock;

    bool parseAndValidateModule(std::span<const uint8_t>);

    const char* stateString(State);
    void moveToState(State);
    bool isComplete() const override { return m_state == State::Completed; }
    void complete() WTF_REQUIRES_LOCK(m_lock) override;

    virtual bool prepareImpl() = 0;
    virtual void compileFunction(FunctionCodeIndex functionIndex) = 0;
    virtual void didCompleteCompilation() WTF_REQUIRES_LOCK(m_lock) = 0;

    template<typename T>
    bool tryReserveCapacity(Vector<T>& vector, size_t size, ASCIILiteral what)
    {
        if (UNLIKELY(!vector.tryReserveCapacity(size))) {
            Locker locker { m_lock };
            fail(WTF::makeString("Failed allocating enough space for "_s, size, what));
            return false;
        }
        return true;
    }

    bool generateWasmToJSStubs();
    bool generateWasmToWasmStubs();

    void generateStubsIfNecessary() WTF_REQUIRES_LOCK(m_lock);

    Vector<uint8_t> m_source;
    Vector<MacroAssemblerCodeRef<WasmEntryPtrTag>> m_wasmToWasmExitStubs;
    Vector<MacroAssemblerCodeRef<WasmEntryPtrTag>> m_wasmToJSExitStubs;
    UncheckedKeyHashSet<uint32_t, DefaultHash<uint32_t>, WTF::UnsignedWithZeroKeyHashTraits<uint32_t>> m_exportedFunctionIndices;

    Vector<Vector<UnlinkedWasmToWasmCall>> m_unlinkedWasmToWasmCalls;
    StreamingParser m_streamingParser;
    State m_state;

    bool m_areWasmToWasmStubsCompiled { false };
    bool m_areWasmToJSStubsCompiled { false };
    const CompilerMode m_compilerMode;
    uint8_t m_numberOfActiveThreads { 0 };
    uint32_t m_currentIndex { 0 };
    uint32_t m_numberOfFunctions { 0 };
};


} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
