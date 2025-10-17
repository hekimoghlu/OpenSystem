/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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

#if ENABLE(JIT)

#include "JITPlan.h"
#include "JITWorklistThread.h"
#include <wtf/Deque.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace JSC {

class CodeBlock;
class VM;

class JITWorklist {
    WTF_MAKE_NONCOPYABLE(JITWorklist);
    WTF_MAKE_TZONE_ALLOCATED(JITWorklist);

    friend class JITWorklistThread;

public:
    ~JITWorklist();

    static JITWorklist& ensureGlobalWorklist();
    static JITWorklist* existingGlobalWorklistOrNull();

    CompilationResult enqueue(Ref<JITPlan>);
    size_t queueLength() const;

    void suspendAllThreads();
    void resumeAllThreads();

    enum State { NotKnown, Compiling, Compiled };
    State compilationState(VM&, JITCompilationKey);

    State completeAllReadyPlansForVM(VM&, JITCompilationKey = JITCompilationKey());

    // This is equivalent to:
    // worklist->waitUntilAllPlansForVMAreReady(vm);
    // worklist->completeAllReadyPlansForVM(vm);
    void completeAllPlansForVM(VM&);

    void cancelAllPlansForVM(VM&);

    void removeDeadPlans(VM&);

    unsigned setMaximumNumberOfConcurrentDFGCompilations(unsigned);
    unsigned setMaximumNumberOfConcurrentFTLCompilations(unsigned);

    // Only called on the main thread after suspending all threads.
    template<typename Visitor>
    void visitWeakReferences(Visitor&);

    template<typename Visitor>
    void iterateCodeBlocksForGC(Visitor&, VM&, const Function<void(CodeBlock*)>&);

    void dump(PrintStream&) const;

private:
    JITWorklist();

    size_t queueLength(const AbstractLocker&) const;

    void waitUntilAllPlansForVMAreReady(VM&);

    template<typename MatchFunction>
    void removeMatchingPlansForVM(VM&, const MatchFunction&);

    State removeAllReadyPlansForVM(VM&, Vector<RefPtr<JITPlan>, 8>&, JITCompilationKey);

    void dump(const AbstractLocker&, PrintStream&) const;

    unsigned m_numberOfActiveThreads { 0 };
    std::array<unsigned, static_cast<size_t>(JITPlan::Tier::Count)> m_ongoingCompilationsPerTier { 0, 0, 0 };
    std::array<unsigned, static_cast<size_t>(JITPlan::Tier::Count)> m_maximumNumberOfConcurrentCompilationsPerTier;

    Vector<Ref<JITWorklistThread>> m_threads;

    // Used to inform the thread about what work there is left to do.
    std::array<Deque<RefPtr<JITPlan>>, static_cast<size_t>(JITPlan::Tier::Count)> m_queues;

    // Used to answer questions about the current state of a code block. This
    // is particularly great for the cti_optimize OSR slow path, which wants
    // to know: did I get here because a better version of me just got
    // compiled?
    UncheckedKeyHashMap<JITCompilationKey, RefPtr<JITPlan>> m_plans;

    // Used to quickly find which plans have been compiled and are ready to
    // be completed.
    Vector<RefPtr<JITPlan>, 16> m_readyPlans;

    Lock m_suspensionLock;
    Box<Lock> m_lock;

    Ref<AutomaticThreadCondition> m_planEnqueued;
    Condition m_planCompiledOrCancelled;
};

} // namespace JSC

#endif // ENABLE(JIT)
