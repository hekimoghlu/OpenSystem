/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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

#include "CompilationResult.h"
#include "DFGDesiredGlobalProperties.h"
#include "DFGDesiredIdentifiers.h"
#include "DFGDesiredTransitions.h"
#include "DFGDesiredWatchpoints.h"
#include "DFGDesiredWeakReferences.h"
#include "DFGFinalizer.h"
#include "DFGJITCode.h"
#include "DeferredCompilationCallback.h"
#include "JITPlan.h"
#include "Operands.h"
#include "ProfilerCompilation.h"
#include "RecordedStatuses.h"
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace JSC {

class CodeBlock;

namespace DFG {

class ThreadData;
class JITData;

#if ENABLE(DFG_JIT)

class Plan final : public JITPlan {
    using Base = JITPlan;

public:
    Plan(
        CodeBlock* codeBlockToCompile, CodeBlock* profiledDFGCodeBlock,
        JITCompilationMode, BytecodeIndex osrEntryBytecodeIndex,
        Operands<std::optional<JSValue>>&& mustHandleValues);
    ~Plan();

    size_t codeSize() const final;
    void finalizeInGC() final;
    CompilationResult finalize() override;

    void notifyReady() final;
    void cancel() final;

    bool isKnownToBeLiveAfterGC() final;
    bool isKnownToBeLiveDuringGC(AbstractSlotVisitor&) final;
    bool iterateCodeBlocksForGC(AbstractSlotVisitor&, const Function<void(CodeBlock*)>&) final;
    bool checkLivenessAndVisitChildren(AbstractSlotVisitor&) final;


    bool canTierUpAndOSREnter() const { return !m_tierUpAndOSREnterBytecodes.isEmpty(); }

    void cleanMustHandleValuesIfNecessary();

    BytecodeIndex osrEntryBytecodeIndex() const { return m_osrEntryBytecodeIndex; }
    const Operands<std::optional<JSValue>>& mustHandleValues() const { return m_mustHandleValues; }
    Profiler::Compilation* compilation() const { return m_compilation.get(); }

    Finalizer* finalizer() const { return m_finalizer.get(); }
    void setFinalizer(std::unique_ptr<Finalizer>&& finalizer) { m_finalizer = WTFMove(finalizer); }

    RefPtr<InlineCallFrameSet> inlineCallFrames() const { return m_inlineCallFrames; }
    DesiredWatchpoints& watchpoints() { return m_watchpoints; }
    DesiredIdentifiers& identifiers() { return m_identifiers; }
    DesiredWeakReferences& weakReferences() { return m_weakReferences; }
    DesiredTransitions& transitions() { return m_transitions; }
    RecordedStatuses& recordedStatuses() { return *m_recordedStatuses.get(); }

    bool willTryToTierUp() const { return m_willTryToTierUp; }
    void setWillTryToTierUp(bool willTryToTierUp) { m_willTryToTierUp = willTryToTierUp; }

    UncheckedKeyHashMap<BytecodeIndex, FixedVector<BytecodeIndex>>& tierUpInLoopHierarchy() { return m_tierUpInLoopHierarchy; }
    Vector<BytecodeIndex>& tierUpAndOSREnterBytecodes() { return m_tierUpAndOSREnterBytecodes; }

    DeferredCompilationCallback* callback() const { return m_callback.get(); }
    void setCallback(Ref<DeferredCompilationCallback>&& callback) { m_callback = WTFMove(callback); }

    std::unique_ptr<JITData> tryFinalizeJITData(const DFG::JITCode&);

private:
    CompilationPath compileInThreadImpl() override;
    void finalizeInThread(Ref<JSC::JITCode>);
    
    bool isStillValidCodeBlock();
    bool reallyAdd(CommonData*);

    // These can be raw pointers because we visit them during every GC in checkLivenessAndVisitChildren.
    CodeBlock* m_profiledDFGCodeBlock;

    Operands<std::optional<JSValue>> m_mustHandleValues;
    bool m_mustHandleValuesMayIncludeGarbage WTF_GUARDED_BY_LOCK(m_mustHandleValueCleaningLock) { true };
    Lock m_mustHandleValueCleaningLock;

    bool m_willTryToTierUp { false };

    const BytecodeIndex m_osrEntryBytecodeIndex;

    RefPtr<Profiler::Compilation> m_compilation;

    std::unique_ptr<Finalizer> m_finalizer;

    RefPtr<InlineCallFrameSet> m_inlineCallFrames;
    DesiredWatchpoints m_watchpoints;
    DesiredIdentifiers m_identifiers;
    DesiredWeakReferences m_weakReferences;
    DesiredTransitions m_transitions;
    std::unique_ptr<RecordedStatuses> m_recordedStatuses;

    UncheckedKeyHashMap<BytecodeIndex, FixedVector<BytecodeIndex>> m_tierUpInLoopHierarchy;
    Vector<BytecodeIndex> m_tierUpAndOSREnterBytecodes;

    RefPtr<DeferredCompilationCallback> m_callback;
};

#endif // ENABLE(DFG_JIT)

} } // namespace JSC::DFG
